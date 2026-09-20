"""Reproducible September retrieval benchmark; notebook and CLI share this runner.

Stages: prepare -> queries -> retrieve -> report. Each stage has explicit disk
artifacts and fingerprints. No answer generation or clinical-case scoring.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

EVAL = Path(__file__).resolve().parents[2]
DEVICE = os.environ.get("SLEEPINN_DEVICE", "cuda")
sys.path.insert(0, str(EVAL / "src"))
os.environ.setdefault("HF_HOME", str(EVAL / "outputs/hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from sleepinn_rag import september_core as core

MODEL = "google/gemma-4-12B-it"
MODEL_REV = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
EMBED = "BAAI/bge-m3"
EMBED_REV = "5617a9f61b028005a4858fdac845db406aefb181"
RERANK = "cross-encoder/ms-marco-MiniLM-L12-v2"
RERANK_REV = "7b0235231ca2674cb8ca8f022859a6eba2b1c968"
SEED = 20260915
DATASET = EVAL / "outputs/private/knowledge_with_evidence.jsonl"
PROMPT = (
    "Generate exactly three alternative search queries for retrieving sleep-medicine reference passages. "
    "Preserve the question's meaning. Use concise medical search phrases. "
    "Do not answer the question or invent patient findings. Return exactly three lines, "
    "one query per line, without numbering, explanations or headings.\n\nQuestion: {question}"
)


def settings():
    return {
        "seed": SEED,
        "dataset_sha256": core.file_digest(DATASET),
        "query_generation_batch_size": 4,
        "model": MODEL,
        "model_revision": MODEL_REV,
        "precision": "bfloat16",
        "embed": EMBED,
        "embed_revision": EMBED_REV,
        "reranker": RERANK,
        "reranker_revision": RERANK_REV,
        "mq_prompt": PROMPT,
        "mq_variants": 3,
        "max_new_tokens": 256,
        "thinking": False,
        "do_sample": False,
        "chunk_size_characters": 1024,
        "chunk_overlap_characters": 512,
        "cleaning": "preserve_pdf_extraction",
        "dense_metric": "cosine_normalized_exact",
        "bm25_tokenization": "whitespace_case_sensitive_legacy",
        "candidates_per_query_per_retriever": 50,
        "rerank_candidates": 50,
        "final_blocks": 8,
        "context_tokens": 4096,
        "rrf_k": 60,
        "rrf_rank_origin": 1,
        "jiwar": {
            "neighbors_each_direction": 1,
            "page_radius": 1,
            "top_anchors": 2,
            "max_constituents": 8,
        },
        "pageagg": {"weights": [1, 0.5, 0.25], "pages": 3},
        "page_convention": "knowledge dataset 0-based; source spans 1-based physical PDF pages",
        "scoring": "compact exact OCR match, then legacy normalized SequenceMatcher threshold 0.82",
        "design": "all_1215_knowledge_items_exploratory_comparison_no_holdout",
        "code_hashes": {
            str(p.relative_to(EVAL)): core.file_digest(p)
            for p in [
                Path(__file__),
                *sorted((EVAL / "src/sleepinn_rag").rglob("*.py")),
            ]
        },
        "pdf_hashes": {
            p.name: core.file_digest(p)
            for p in sorted((EVAL / "outputs/private/sources/knowledge").glob("*.pdf"))
        },
    }


def log_status(run, stage, **extra):
    record = {
        "stage": stage,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **extra,
    }
    core.atomic_json(run / "status.json", record)
    print(json.dumps(record), flush=True)


def initialize(run):
    if not DATASET.exists():
        raise FileNotFoundError("Retrieval scoring needs licensed PDFs and the evidence-enriched bank; see REPRODUCING.md")
    import torch

    if not DEVICE.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError(
            "The frozen retrieval runner requires a BF16-capable CUDA device"
        )
    torch.cuda.set_device(torch.device(DEVICE))
    run.mkdir(parents=True, exist_ok=True)
    params = settings()
    fingerprint = core.digest(params)
    manifest_path = run / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["fingerprint"] != fingerprint:
            raise RuntimeError(
                "Run settings/source/code changed. Use a new run ID; do not reuse stale caches."
            )
    else:
        versions = {}
        for pkg in [
            "torch",
            "transformers",
            "sentence-transformers",
            "pandas",
            "numpy",
            "rank-bm25",
            "pymupdf",
        ]:
            try:
                versions[pkg] = importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError:
                versions[pkg] = None
        gpu = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
            ).stdout.strip()
            if __import__("shutil").which("nvidia-smi")
            else "unavailable"
        )
        core.atomic_json(
            manifest_path,
            {
                "fingerprint": fingerprint,
                "settings": params,
                "packages": versions,
                "gpu": gpu,
                "python": sys.version,
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )
        core.atomic_json(
            run / "strategy_matrix.json", core.strategy_matrix(EVAL / "src")
        )
        import shutil

        for name in params["code_hashes"]:
            destination = run / "source_code" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(EVAL / name, destination)
    items = [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    items = [item for item in items if item["item_type"] != "clinical_case"]
    assert len(items) == 1215 and len({i["item_id"] for i in items}) == 1215
    assert all(i.get("gold_evidence_units") for i in items)
    return items


def prepare(run, items):
    import fitz
    import numpy as np
    import torch
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from sleepinn_rag.kb.scope import (
        extract_document_structure,
        infer_chunk_role,
        build_contextualized_chunk,
    )
    from sentence_transformers import SentenceTransformer

    started = time.perf_counter()
    log_status(run, "extracting_and_indexing")
    pages = []
    sources = sorted({i["source_pdf"] for i in items})
    assert sources == sorted(p.name for p in (EVAL / "outputs/private/sources/knowledge").glob("*.pdf"))
    for source in sources:
        with fitz.open(EVAL / "outputs/private/sources/knowledge" / source) as pdf:
            for page_index, page in enumerate(pdf):
                pages.append(
                    Document(
                        page_content=page.get_text().strip(),
                        metadata={"source": source, "page": page_index + 1},
                    )
                )
    page_text = {
        (p.metadata["source"], p.metadata["page"]): p.page_content for p in pages
    }
    core.atomic_json(
        run / "pages.json",
        [
            {"source_pdf": source, "page": pg, "text": text}
            for (source, pg), text in page_text.items()
        ],
    )
    structures = extract_document_structure(pages)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=512,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True,
    )
    chunks = []
    for raw in splitter.split_documents(pages):
        source, page, start = (
            raw.metadata["source"],
            raw.metadata["page"],
            raw.metadata["start_index"],
        )
        assert (
            start >= 0
            and page_text[(source, page)][start : start + len(raw.page_content)]
            == raw.page_content
        )
        structure = structures[(source, page)]
        chunks.append(
            {
                "index": len(chunks),
                "source_pdf": source,
                "page": page,
                "start": start,
                "end": start + len(raw.page_content),
                "text": raw.page_content,
                "scope_text": build_contextualized_chunk(
                    raw.page_content,
                    page=page,
                    chunk_type=infer_chunk_role(raw.page_content),
                    **structure,
                ),
            }
        )
    assert chunks == sorted(
        chunks, key=lambda c: (c["source_pdf"], c["page"], c["start"])
    )
    core.atomic_json(run / "chunks.json", chunks)
    print("Raw chunk inventory", len(chunks), "pages", len(pages), flush=True)
    torch.set_num_threads(4)
    model = SentenceTransformer(EMBED, revision=EMBED_REV, device=DEVICE)
    for field, name in [("text", "standard"), ("scope_text", "scope")]:
        vectors = model.encode(
            [c[field] for c in chunks],
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        np.save(run / f"{name}_embeddings.npy", vectors.astype("float32"))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    availability = []
    for item in items:
        matches = []
        for i, unit in enumerate(item["gold_evidence_units"]):
            texts = [
                page_text[(item["source_pdf"], pg)]
                for pg in sorted(core.gold_pages(item, i))
            ]
            matches.append(
                any(core.evidence_match(unit, text) for text in texts)
                or core.evidence_match(unit, "\n".join(texts))
            )
        availability.append(
            {
                "item_id": item["item_id"],
                "source_pdf": item["source_pdf"],
                "gold_units": len(matches),
                "available_units": sum(matches),
                "units_match": matches,
            }
        )
    core.atomic_json(run / "source_evidence_audit.json", availability)
    elapsed = time.perf_counter() - started
    core.atomic_json(
        run / "preparation.json",
        {
            "seconds": elapsed,
            "chunks": len(chunks),
            "pages": len(pages),
            "items_with_all_source_evidence": sum(
                all(r["units_match"]) for r in availability
            ),
            "items": len(items),
            "gold_units": sum(r["gold_units"] for r in availability),
            "available_gold_units": sum(r["available_units"] for r in availability),
        },
    )
    log_status(run, "prepared", seconds=elapsed, chunks=len(chunks))


def selected_items(items, limit):
    if not limit:
        return items
    groups = {}
    rng = random.Random(SEED)
    for item in items:
        key = (
            item["source_pdf"],
            item.get("metadata", {}).get(
                "chapter_id", item.get("section_label", "unknown")
            ),
            item["item_type"],
        )
        groups.setdefault(key, []).append(item)
    for group in groups.values():
        rng.shuffle(group)
    keys = sorted(groups)
    # Both sources first; then cycle across their chapter/type groups.
    ordered_keys = []
    sources = sorted({key[0] for key in keys})
    by_source = {src: [key for key in keys if key[0] == src] for src in sources}
    while any(by_source.values()):
        for source in sources:
            if by_source[source]:
                ordered_keys.append(by_source[source].pop(0))
    chosen = []
    while len(chosen) < min(limit, len(items)):
        for key in ordered_keys:
            if groups[key] and len(chosen) < limit:
                chosen.append(groups[key].pop())
    return chosen


def final_response_text(parsed):
    if isinstance(parsed, str):
        return parsed
    if isinstance(parsed, dict):
        if isinstance(parsed.get("content"), str):
            return parsed["content"]
        if isinstance(parsed.get("content"), list):
            return "\n".join(
                part["text"]
                for part in parsed["content"]
                if isinstance(part, dict)
                and "text" in part
                and part.get("type", "text") == "text"
                and part.get("channel", "final") != "analysis"
            )
        if isinstance(parsed.get("text"), str):
            return parsed["text"]
    raise ValueError("Unrecognized Gemma final response format")


def generate_queries(run, items, args):
    import torch
    from transformers import AutoProcessor, AutoModelForMultimodalLM

    dictionaries = core.load_dictionaries(EVAL / "src")
    chosen = selected_items(items, args.limit)
    query_dir = run / "queries"
    query_dir.mkdir(exist_ok=True)
    pending = [
        item for item in chosen if not (query_dir / f"{item['item_id']}.json").exists()
    ]
    if args.batch_size != 4:
        raise ValueError(
            "This frozen experiment uses batches of four independent Gemma prompts."
        )
    if not pending:
        log_status(run, "queries_ready", items=len(chosen), reused=True)
        return
    log_status(run, "loading_gemma", remaining=len(pending))
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(MODEL, revision=MODEL_REV)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL,
        revision=MODEL_REV,
        dtype=torch.bfloat16,
        device_map={"": DEVICE},
        attn_implementation="sdpa",
    )
    model.eval()
    torch.manual_seed(SEED)
    load_seconds = time.perf_counter() - started
    generation_started = time.perf_counter()
    timings = []

    def generate(prompts):
        conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
        inputs = processor.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
            add_generation_prompt=True,
            enable_thinking=False,
        ).to(DEVICE)
        torch.cuda.synchronize()
        stamp = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - stamp
        records = []
        for index in range(len(prompts)):
            token_ids = outputs[index][inputs["input_ids"].shape[-1] :].tolist()
            trimmed = core.trim_generation_padding(
                token_ids, processor.tokenizer.pad_token_id
            )
            raw = processor.decode(token_ids, skip_special_tokens=False)
            parser_input = processor.decode(trimmed, skip_special_tokens=False)
            parsed = processor.parse_response(
                parser_input, prefix=inputs["input_ids"][index]
            )
            records.append(
                {
                    "raw": raw,
                    "parser_input": parser_input,
                    "padding_tokens_removed": len(token_ids) - len(trimmed),
                    "parsed": parsed,
                    "final_text": final_response_text(parsed),
                }
            )
        return records, elapsed

    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset : offset + args.batch_size]
        originals = [
            core.expand_abbreviations(item["question"], dictionaries["ABBREVIATIONS"])
            for item in batch
        ]
        prompts = [PROMPT.format(question=q) for q in originals]
        responses, seconds = generate(prompts)
        timings.append({"items": len(batch), "seconds": seconds})
        for item, original, prompt, response in zip(
            batch, originals, prompts, responses
        ):
            attempts = [{"prompt": prompt, **response}]
            attempt_path = run / "query_attempts" / f"{item['item_id']}.json"
            core.atomic_json(
                attempt_path, {"item_id": item["item_id"], "attempts": attempts}
            )
            additional_seconds = 0.0
            try:
                queries = core.parse_queries(response["final_text"], original)
            except ValueError:
                retry_prompt = (
                    prompt
                    + "\nUse three distinct queries, each different from the original question. Output only the three lines."
                )
                retry, additional_seconds = generate([retry_prompt])
                attempts.append({"prompt": retry_prompt, **retry[0]})
                core.atomic_json(
                    attempt_path, {"item_id": item["item_id"], "attempts": attempts}
                )
                queries = core.parse_queries(retry[0]["final_text"], original)
            core.atomic_json(
                query_dir / f"{item['item_id']}.json",
                {
                    "item_id": item["item_id"],
                    "question_sha256": core.digest(item["question"]),
                    "expanded_question": original,
                    "queries": queries,
                    "generation_batch_size": len(batch),
                    "generation_batch_seconds": seconds,
                    "amortized_generation_seconds": seconds / len(batch)
                    + additional_seconds,
                    "retry_count": len(attempts) - 1,
                    "attempts": attempts,
                },
            )
        completed = min(offset + len(batch), len(pending))
        elapsed = time.perf_counter() - generation_started
        if completed == len(pending) or completed <= 8 or completed % 20 == 0:
            log_status(
                run,
                "generating_queries",
                completed=completed,
                pending_at_start=len(pending),
                elapsed_seconds=elapsed,
                estimated_remaining_seconds=elapsed
                / completed
                * (len(pending) - completed),
            )
    core.atomic_json(
        run / ("query_pilot_timing.json" if args.limit else "query_timing.json"),
        {
            "model_load_seconds": load_seconds,
            "batch_timings": timings,
            "wall_seconds": time.perf_counter() - generation_started,
            "generated_items": len(pending),
            "batch_size": args.batch_size,
            "max_gpu_memory_GB": torch.cuda.max_memory_allocated() / 1e9,
        },
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    log_status(run, "queries_ready", items=len(chosen))


def retrieve(run, items, args):
    import numpy as np
    import torch
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from transformers import AutoProcessor

    started = time.perf_counter()
    chosen = selected_items(items, args.limit)
    matrix = core.strategy_matrix(EVAL / "src")
    dictionaries = core.load_dictionaries(EVAL / "src")
    chunks = json.loads((run / "chunks.json").read_text())
    pages = {
        (p["source_pdf"], p["page"]): p["text"]
        for p in json.loads((run / "pages.json").read_text())
    }
    vectors = {
        False: np.load(run / "standard_embeddings.npy"),
        True: np.load(run / "scope_embeddings.npy"),
    }
    sparse = {
        scope: BM25Okapi([c["scope_text" if scope else "text"].split() for c in chunks])
        for scope in [False, True]
    }
    queries = {
        i["item_id"]: json.loads((run / "queries" / f"{i['item_id']}.json").read_text())
        for i in chosen
    }
    for item in chosen:
        assert queries[item["item_id"]]["question_sha256"] == core.digest(
            item["question"]
        )
    embedder = SentenceTransformer(EMBED, revision=EMBED_REV, device=DEVICE)
    reranker = CrossEncoder(RERANK, revision=RERANK_REV, device=DEVICE, max_length=512)
    tokenizer = AutoProcessor.from_pretrained(MODEL, revision=MODEL_REV).tokenizer
    torch.set_num_threads(4)
    outdir = run / "items"
    outdir.mkdir(exist_ok=True)
    pending = [
        item for item in chosen if not (outdir / f"{item['item_id']}.json").exists()
    ]
    # Embeddings are shared by all configurations, with measured amortized throughput cost.
    query_texts = core.deduplicate(
        q
        for item in pending
        for q in [
            item["question"],
            queries[item["item_id"]]["expanded_question"],
            *queries[item["item_id"]]["queries"],
        ]
    )
    if not query_texts:
        log_status(run, "retrieval_ready", items=len(chosen), reused=True)
        return
    log_status(
        run,
        "embedding_questions",
        unique_query_texts=len(query_texts),
        pending_items=len(pending),
    )
    torch.cuda.synchronize()
    stamp = time.perf_counter()
    q_vectors = embedder.encode(
        query_texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True
    )
    torch.cuda.synchronize()
    embedding_seconds = time.perf_counter() - stamp
    query_embeddings = dict(zip(query_texts, q_vectors))
    embedding_seconds_each = embedding_seconds / len(query_texts)
    del embedder
    gc.collect()
    torch.cuda.empty_cache()
    load_seconds = time.perf_counter() - started
    retrieval_started = time.perf_counter()
    item_times = []
    for item_number, item in enumerate(pending):
        item_started = time.perf_counter()
        query_info = queries[item["item_id"]]
        search_cache, score_cache = {}, {}
        plans = []
        arm_order = matrix[:]
        random.Random(SEED + item_number).shuffle(arm_order)
        for flags in arm_order:
            original = (
                query_info["expanded_question"]
                if flags["abbreviations"]
                else item["question"]
            )
            # Original first gives the single-query baseline full representation in non-RRF MQ unions.
            qlist = core.deduplicate(
                [original] + (query_info["queries"] if flags["mq"] else [])
            )
            dense_lists, bm25_lists, search_keys = [], [], []
            for q in qlist:
                for dense in [True, False]:
                    if not flags["dense" if dense else "bm25"]:
                        continue
                    used_query = (
                        q
                        if dense or not flags["bm25_terms"]
                        else core.expand_terms(q, dictionaries["CLINICAL_SYNONYMS"])
                    )
                    key = (flags["scope"], dense, used_query)
                    if key not in search_cache:
                        stamp = time.perf_counter()
                        values = (
                            vectors[flags["scope"]] @ query_embeddings[q]
                            if dense
                            else sparse[flags["scope"]].get_scores(used_query.split())
                        )
                        order = np.argsort(-values, kind="stable")[:50]
                        search_cache[key] = {
                            "indices": order.astype(int).tolist(),
                            "scores": values[order].astype(float).tolist(),
                            "seconds": time.perf_counter() - stamp,
                            "query_embedding_seconds_amortized": embedding_seconds_each
                            if dense
                            else 0.0,
                        }
                    (dense_lists if dense else bm25_lists).append(
                        search_cache[key]["indices"]
                    )
                    search_keys.append(key)
            stamp = time.perf_counter()
            fused, fusion_scores = core.fuse_lists(dense_lists, bm25_lists, flags)
            plans.append(
                {
                    "flags": flags,
                    "original": original,
                    "queries": qlist,
                    "search_keys": search_keys,
                    "fused": fused,
                    "fusion_scores": fusion_scores,
                    "fusion_seconds": time.perf_counter() - stamp,
                }
            )
        # Score each unique (query, representation, candidate) once for all reranker arms.
        pairs = core.deduplicate(
            (p["original"], p["flags"]["scope"], index)
            for p in plans
            if p["flags"]["rerank"]
            for index in p["fused"][:50]
        )
        torch.cuda.synchronize()
        stamp = time.perf_counter()
        scores = reranker.predict(
            [
                [q, chunks[index]["scope_text" if scope else "text"]]
                for q, scope, index in pairs
            ],
            batch_size=64,
            show_progress_bar=False,
        )
        torch.cuda.synchronize()
        rerank_seconds = time.perf_counter() - stamp
        score_cache.update({key: float(score) for key, score in zip(pairs, scores)})
        result_rows = []
        for plan in plans:
            flags = plan["flags"]
            ranking = plan["fused"]
            page_scores = []
            if flags["rerank"]:
                ranked_scores = sorted(
                    [
                        (index, score_cache[(plan["original"], flags["scope"], index)])
                        for index in ranking[:50]
                    ],
                    key=lambda pair: (-pair[1], pair[0]),
                )
                ranking = [index for index, _ in ranked_scores]
                if flags["pageagg"]:
                    ranking, page_scores = core.aggregate_pages(ranked_scores, chunks)
            anchors = ranking[:8]
            stamp = time.perf_counter()
            raw_blocks = core.make_blocks(anchors, chunks, pages, jiwar=flags["jiwar"])
            budgeted, token_count = core.apply_budget(raw_blocks, tokenizer, 4096)
            assembly_seconds = time.perf_counter() - stamp
            metrics = core.score_blocks(item, raw_blocks)
            metrics.update(core.score_blocks(item, budgeted, prefix="budgeted_"))
            anchor_metrics = core.score_blocks(
                item, core.make_blocks(anchors, chunks, pages)
            )
            component_search_seconds = sum(
                search_cache[key]["seconds"]
                + search_cache[key]["query_embedding_seconds_amortized"]
                for key in plan["search_keys"]
            )
            component_rerank_seconds = (
                rerank_seconds / len(pairs) * min(50, len(plan["fused"]))
                if flags["rerank"] and pairs
                else 0.0
            )
            mq_seconds = (
                query_info["amortized_generation_seconds"] if flags["mq"] else 0.0
            )
            # This is a throughput-cost estimate, deliberately not labeled isolated cold request latency.
            estimated_seconds = (
                component_search_seconds
                + component_rerank_seconds
                + mq_seconds
                + plan["fusion_seconds"]
                + assembly_seconds
            )
            result_rows.append(
                {
                    "strategy": flags["strategy"],
                    "item_id": item["item_id"],
                    "item_type": item["item_type"],
                    "source_pdf": item["source_pdf"],
                    "chapter": item.get("metadata", {}).get(
                        "chapter_id", item.get("section_label", "unknown")
                    ),
                    "gold_evidence_units": len(item["gold_evidence_units"]),
                    "status": "ok",
                    "context_tokens": token_count,
                    "num_blocks": len(budgeted),
                    **metrics,
                    "anchor_evidence_recall@5": anchor_metrics["evidence_recall@5"],
                    "search_seconds_amortized": component_search_seconds,
                    "rerank_seconds_amortized": component_rerank_seconds,
                    "mq_seconds_amortized": mq_seconds,
                    "assembly_seconds": assembly_seconds,
                    "estimated_component_cost_seconds": estimated_seconds,
                    "queries": plan["queries"],
                    "fused_candidates": plan["fused"][:50],
                    "fusion_scores": plan["fusion_scores"],
                    "reranked_candidates": ranking[:50],
                    "page_scores": page_scores,
                    "unbudgeted_blocks": raw_blocks,
                    "delivered_blocks": budgeted,
                }
            )
        item_elapsed = time.perf_counter() - item_started
        item_times.append(item_elapsed)
        core.atomic_json(
            outdir / f"{item['item_id']}.json",
            {
                "item_id": item["item_id"],
                "query_info_file": f"queries/{item['item_id']}.json",
                "item_wall_seconds": item_elapsed,
                "searches": [
                    {"scope": key[0], "dense": key[1], "query": key[2], **value}
                    for key, value in search_cache.items()
                ],
                "reranker_scores": [
                    {
                        "query": key[0],
                        "scope": key[1],
                        "chunk_index": key[2],
                        "score": value,
                    }
                    for key, value in score_cache.items()
                ],
                "rows": result_rows,
            },
        )
        core.evidence_match.cache_clear()
        completed = item_number + 1
        if completed <= 3 or completed % 10 == 0 or completed == len(pending):
            elapsed = time.perf_counter() - retrieval_started
            log_status(
                run,
                "retrieving",
                completed=completed,
                pending_at_start=len(pending),
                elapsed_seconds=elapsed,
                estimated_remaining_seconds=elapsed
                / completed
                * (len(pending) - completed),
            )
    core.atomic_json(
        run
        / ("retrieval_pilot_timing.json" if args.limit else "retrieval_timing.json"),
        {
            "load_and_query_embedding_seconds": load_seconds,
            "query_embedding_seconds": embedding_seconds,
            "items_processed": len(pending),
            "item_wall_seconds": item_times,
            "retrieval_wall_seconds": time.perf_counter() - retrieval_started,
            "timing_interpretation": "Shared components and GPU batches; cost columns are amortized estimates, not isolated per-request latency.",
        },
    )
    log_status(run, "retrieval_ready", items=len(chosen), strategies=len(matrix))


def report(run, items):
    import pandas as pd

    records = []
    for path in sorted((run / "items").glob("*.json")):
        records.extend(json.loads(path.read_text())["rows"])
    if not records:
        raise RuntimeError("No completed retrieval records")
    omitted = {
        "queries",
        "fused_candidates",
        "fusion_scores",
        "reranked_candidates",
        "page_scores",
        "unbudgeted_blocks",
        "delivered_blocks",
    }
    frame = pd.DataFrame(
        [
            {key: value for key, value in row.items() if key not in omitted}
            for row in records
        ]
    )
    assert not frame.duplicated(["item_id", "strategy"]).any()
    expected_arms = {r["strategy"] for r in core.strategy_matrix(EVAL / "src")}
    assert all(
        set(group.strategy) == expected_arms for _, group in frame.groupby("item_id")
    )
    frame.to_csv(run / "retrieval_benchmark.csv", index=False)
    metric_columns = [c for c in frame if "recall@" in c or "mrr" in c or "hit@" in c]
    summary = frame.groupby("strategy")[
        metric_columns + ["context_tokens", "estimated_component_cost_seconds"]
    ].mean()
    summary["items"] = frame.groupby("strategy").size()
    summary = summary.sort_values(
        "budgeted_evidence_recall@5", ascending=False, kind="stable"
    )
    summary.to_csv(run / "summary_overall.csv")
    for column in ["item_type", "source_pdf", "chapter"]:
        grouped = frame.groupby(["strategy", column])[metric_columns].mean()
        grouped["items"] = frame.groupby(["strategy", column]).size()
        grouped.to_csv(run / f"summary_by_{column}.csv")
    complete = frame.item_id.nunique() == len(items)
    core.atomic_json(
        run / "completion.json",
        {
            "complete": complete,
            "items": int(frame.item_id.nunique()),
            "expected_items": len(items),
            "strategies": len(expected_arms),
            "rows": len(frame),
            "winner_by_full_dataset_budgeted_recall5": summary.index[0],
            "interpretation": "Exploratory full-dataset comparison; no independent confirmation subset. "
            "Component-cost estimates are not isolated latency measurements.",
        },
    )
    log_status(
        run,
        "complete" if complete else "pilot_complete",
        items=int(frame.item_id.nunique()),
        rows=len(frame),
    )
    print(
        summary[["budgeted_evidence_recall@5", "evidence_recall@5", "items"]]
        .head(10)
        .to_string(),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="retrieval_new_001")
    parser.add_argument(
        "--stage",
        choices=["prepare", "queries", "retrieve", "report", "all"],
        default="prepare",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stratified pilot item count; zero means all knowledge items",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if not args.run_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Invalid run ID")
    run = EVAL / "outputs/retrieval" / args.run_id
    try:
        items = initialize(run)
        if args.stage == "prepare" or (
            args.stage == "all" and not (run / "preparation.json").exists()
        ):
            prepare(run, items)
        if args.stage in ("queries", "all"):
            generate_queries(run, items, args)
        if args.stage in ("retrieve", "all"):
            retrieve(run, items, args)
        if args.stage in ("report", "all"):
            report(run, items)
    except Exception:
        log_status(run, "failed", traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
