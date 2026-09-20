"""Run frozen prompts with pinned local checkpoints on a chosen CUDA device."""
from pathlib import Path
import gc
import json
import os
import shutil
import time

from .io import digest, file_hash, project_root, read_json, read_jsonl, write_json


def model_registry():
    return read_json(project_root() / "data/config/models.json")


def prompt_jobs(domain, conditions=None):
    folder = project_root() / "outputs/prompts" / domain
    if not (folder / "paired_prompts.jsonl").exists():
        raise FileNotFoundError("Prepare source-based prompts first; see notebook 05 and REPRODUCING.md")
    prompts = read_jsonl(folder / "paired_prompts.jsonl")
    pilot = read_json(folder / "pilot_item_ids.json")
    if conditions is None:
        conditions = (
            ["no_rag", "with_rag", "perfect_retrieval"]
            if domain == "knowledge"
            else ["no_rag", "with_rag"]
        )
    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("Choose distinct, nonempty conditions")
    by_id = {row["item_id"]: row for row in prompts}
    if len(by_id) != len(prompts):
        raise ValueError("Duplicate prompt identities")
    order = pilot + [row["item_id"] for row in prompts if row["item_id"] not in pilot]
    if len(order) != len(set(order)) or set(order) != set(by_id):
        raise ValueError("Invalid question order")
    jobs = []
    for index, identity in enumerate(order):
        row = by_id[identity]
        shift = index % len(conditions)
        for mode in conditions[shift:] + conditions[:shift]:
            prompt = row["prompts"][mode]
            if digest(prompt.encode()) != row["prompt_sha256"][mode]:
                raise ValueError("Frozen prompt hash mismatch")
            jobs.append(
                dict(
                    dataset=domain,
                    item_id=identity,
                    mode=mode,
                    prompt=prompt,
                    prompt_sha256=row["prompt_sha256"][mode],
                )
            )
    return jobs


def _spark_checkpoint(entry, cache):
    """Apply the documented activation-cast correction to a separate checkpoint view."""
    from huggingface_hub import snapshot_download

    original = Path(
        snapshot_download(
            entry["model_id"],
            revision=entry["revision"],
            cache_dir=cache,
            allow_patterns=[
                "*.json",
                "*.py",
                "*.safetensors",
                "*.model",
                "*.jinja",
                "*.txt",
            ],
        )
    )
    code = (original / "modeling_spark.py").read_bytes()
    expected = "9cf0d1ad2b54b9f7088792779ddd8b5cb4d4fe63bfea054da7f2dd16362bcaf4"
    if digest(code) != expected:
        raise ValueError("SPARK source differs from the audited revision")
    before = "hidden_states = hidden_states.to(self.mlp.gate_proj.weight.dtype)"
    after = 'hidden_states = hidden_states.to(getattr(self.mlp.gate_proj, "compute_dtype", None) or self.mlp.gate_proj.weight.dtype)'
    text = code.decode()
    if text.count(before) != 2:
        raise ValueError("Unexpected SPARK activation casts")
    adapted = cache / "spark-adapted" / entry["revision"]
    adapted.mkdir(parents=True, exist_ok=True)
    patched = text.replace(before, after).encode()
    for source in original.iterdir():
        target = adapted / source.name
        if source.name == "modeling_spark.py":
            if target.exists() and target.read_bytes() != patched:
                raise ValueError("Adapted SPARK source has changed")
            target.write_bytes(patched)
        elif source.is_file() and not target.exists():
            try:
                os.link(source.resolve(), target)
            except OSError:
                shutil.copy2(source, target)
    return adapted, {"original_sha256": digest(code), "adapted_sha256": digest(patched)}


def run_local(model_id, domain, precision, destination, *, device="cuda:0", limit=None):
    """Use a new output folder. No precision fallback, CPU offload, or implicit retry."""
    import torch
    import transformers
    from transformers import (
        AutoTokenizer,
        AutoProcessor,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
    )
    import bitsandbytes as bnb

    entry = next(item for item in model_registry() if item["model_id"] == model_id)
    if precision not in entry.get(domain + "_precisions", []):
        raise ValueError(
            "This model/domain/precision is outside the included study scope"
        )
    if transformers.__version__ != entry["transformers"]:
        raise RuntimeError(
            "Use Transformers " + entry["transformers"] + " in this model's environment"
        )
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError(
            "Historical NF4/BF16 inference requires a BF16-capable CUDA device"
        )
    cuda = torch.device(device if ":" in device else f"cuda:{torch.cuda.current_device()}")
    torch.cuda.set_device(cuda)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This device does not support BF16")
    destination = Path(destination).resolve()
    if not destination.is_relative_to((project_root() / "outputs").resolve()):
        raise ValueError("New benchmarks belong under outputs/")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            "Use a new empty run folder; completed data are never overwritten"
        )
    destination.mkdir(parents=True, exist_ok=True)
    jobs = prompt_jobs(domain)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        jobs = jobs[:limit]
    protocol = dict(
        model_id=model_id,
        revision=entry["revision"],
        dataset=domain,
        precision=precision,
        seed=20260915,
        max_new_tokens=1024,
        do_sample=False,
        enable_thinking=False,
        batch_size=1,
        device=device,
        torch=torch.__version__,
        transformers=transformers.__version__,
        bitsandbytes=bnb.__version__,
        gpu=torch.cuda.get_device_name(cuda),
        prompt_file_sha256=file_hash(
            project_root() / f"outputs/prompts/{domain}/paired_prompts.jsonl"
        ),
        worker_sha256=file_hash(Path(__file__)),
        jobs=len(jobs),
        release="portable-1.0; a new deployment, not the historical execution",
    )
    write_json(destination / "protocol.json", protocol)
    cache = project_root() / "outputs/hf_cache"
    cache.mkdir(parents=True, exist_ok=True)
    custom = "Spark" in model_id
    source = model_id
    kwargs = dict(revision=entry["revision"], trust_remote_code=custom, cache_dir=cache)
    if custom:
        source, adaptation = _spark_checkpoint(entry, cache)
        kwargs.pop("revision")
        write_json(destination / "spark_adaptation.json", adaptation)
    torch.manual_seed(protocol["seed"])
    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    multimodal = "ConditionalGeneration" in entry["architecture"][0]
    processor = (
        AutoProcessor.from_pretrained(source, **kwargs) if multimodal else tokenizer
    )
    model_class = (
        getattr(transformers, entry["architecture"][0])
        if multimodal
        else AutoModelForCausalLM
    )
    quantization = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        if precision == "NF4"
        else None
    )
    if (
        precision == "BF16"
        and entry["safetensors"]["total"] * 2 + 4 * 1024**3
        >= torch.cuda.get_device_properties(cuda).total_memory
    ):
        raise RuntimeError("Insufficient memory for BF16 weights plus working space")
    model = model_class.from_pretrained(
        source,
        **kwargs,
        dtype=torch.bfloat16,
        quantization_config=quantization,
        device_map={"": str(cuda)},
        attn_implementation="eager" if custom else "sdpa",
    )
    try:
        model.eval()
        quantized = sum(
            isinstance(module, bnb.nn.Linear4bit) for module in model.modules()
        )
        if (precision == "NF4") != bool(quantized):
            raise RuntimeError("Loaded precision differs from the requested precision")
        exceptions = [
            {"name": name, "dtype": str(parameter.dtype), "numel": parameter.numel()}
            for name, parameter in model.named_parameters()
            if parameter.numel() >= 1_000_000
            and not isinstance(parameter, bnb.nn.Params4bit)
        ]
        if precision == "NF4" and any("expert" in row["name"] for row in exceptions):
            raise RuntimeError("Large expert tensors remain unquantized")
        if precision == "BF16" and any(
            row["dtype"] != "torch.bfloat16" for row in exceptions
        ):
            raise RuntimeError("Large weights are not uniformly BF16")
        if any(parameter.device != cuda for parameter in model.parameters()):
            raise RuntimeError("Unexpected device placement or offload")
        write_json(
            destination / "model_manifest.json",
            {
                "linear4bit_modules": quantized,
                "large_non_4bit_parameters": exceptions,
                "generation_config": model.generation_config.to_dict(),
            },
        )

        def encode(prompt):
            return processor.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=False,
            )

        config = (
            model.config.get_text_config()
            if hasattr(model.config, "get_text_config")
            else model.config
        )
        context_limit = getattr(
            config, "max_position_embeddings", tokenizer.model_max_length
        )
        if not isinstance(context_limit, int) or not 0 < context_limit < 10**9:
            raise ValueError("No finite context limit")
        for job in jobs:
            if encode(job["prompt"])["input_ids"].shape[-1] + 1024 > context_limit:
                raise ValueError("Context overflow; prompt truncation is disabled")
        pad = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        with torch.inference_mode():
            model.generate(
                **encode("Reply with the word Ready.").to(cuda),
                max_new_tokens=16,
                do_sample=False,
                use_cache=True,
                pad_token_id=pad,
            )
        for index, job in enumerate(jobs):
            inputs = encode(job["prompt"]).to(cuda)
            length = inputs["input_ids"].shape[-1]
            torch.cuda.synchronize(cuda)
            torch.cuda.reset_peak_memory_stats(cuda)
            started = time.perf_counter()
            with torch.inference_mode():
                tokens = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=pad,
                )
            torch.cuda.synchronize(cuda)
            elapsed = time.perf_counter() - started
            generated = tokens[0, length:]
            answer = tokenizer.decode(generated, skip_special_tokens=True)
            result = {key: value for key, value in job.items() if key != "prompt"}
            result.update(
                model_id=model_id,
                revision=entry["revision"],
                precision=precision,
                answer=answer,
                answer_sha256=digest(answer.encode()),
                input_tokens=length,
                output_tokens=len(generated),
                max_new_tokens_reached=len(generated) >= 1024,
                generation_seconds=elapsed,
                peak_allocated_gib=torch.cuda.max_memory_allocated(cuda) / 1024**3,
                peak_reserved_gib=torch.cuda.max_memory_reserved(cuda) / 1024**3,
                output_tokens_per_second=len(generated) / elapsed,
            )
            write_json(destination / "rows" / f"{index:05d}.json", result)
        write_json(destination / "completion.json", {"answers": len(jobs)})
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
