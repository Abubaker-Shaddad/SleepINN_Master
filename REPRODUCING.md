# Reproducing the study

## Offline analysis

Install `requirements-analysis.txt` and the package, then run the numbered notebooks. The public release includes the questions, references, model answers, question-level labels, paired membership, summary tables, retrieval measurements and observed GPU-memory data. No API access is required to reproduce the reported analyses.

```bash
python -m unittest discover -s tests
python -m sleepinn_study.audit
python -m sleepinn_study.viewers
```

The last command rebuilds `results/dataset_viewer.html` and `results/model_results_viewer.html`. These self-contained files need no server, account or external JavaScript library.

## New dataset creation and human review

Notebook 01 shows source extraction, historical ICSD generation instructions and reusable prompts for new batches. Exact historical AASM API payloads are not supplied; new helper prompts are not presented as the original calls. Notebook 02 shows the AI revision instructions and a human correction form. Notebook 03 covers clinical authoring, its human correction form and final assembly.

Review forms save decisions under `outputs/human_review/`. Use a reviewer identifier rather than a personal name if the logs may be shared. They never replace the released databank. A final export is blocked until every item has an explicit decision. `assemble_final` verifies both export manifests before creating a combined bank.

The supplied release-level expert-review statement records what the author reported. It is not an item-level historical audit. The included interface can produce such an audit for future revisions.

## Retrieval and source-dependent generation

Bring your own licensed source PDFs and retain their exact filenames:

```text
outputs/private/sources/knowledge/     ICSD-3-TR and AASM PDFs
outputs/private/knowledge_with_evidence.jsonl
outputs/private/knowledge_contexts.json
outputs/private/knowledge_perfect_contexts.json
outputs/private/clinical_contexts.json
```

The evidence-enriched JSONL needs the knowledge fields plus `gold_evidence_units` containing aligned source quotations. The released projection deliberately omits those quotations. A local source/evidence review is required before rerunning evidence recall; page ranges alone do not reconstruct the original evidence units. Exact reconstruction of the historical source-dependent generation is therefore not possible from the public projection alone.

The retrieval runner preserves the study's chunking, ranking, reranking and packing operations. Its `prepare` stage builds source spans, standard and SCOPE embeddings. `queries` creates the Gemma query variants; `retrieve` evaluates the configurations and saves their delivered blocks; `report` aggregates the measurements.

```bash
python -m sleepinn_study.retrieval_experiment --run-id retrieval_new_001 --stage all
```

For a new answer experiment, create maps from item IDs to the packed source-context strings, using the selected configuration's delivered blocks. Keep perfect-retrieval contexts separate. Notebook 05 freezes the prompts and their hashes once before inference. Historical prompt hashes are distributed for comparison; context reconstruction or changes in tokenizer/runtime may produce a new experiment rather than an exact replay.

## Model environments

Analysis can run on a CPU. Fresh inference requires a compatible CUDA GPU and sufficient memory. The study used NVIDIA workstation GPUs and an 80 GiB A100 for larger BF16 models; execution code accepts a CUDA device without rental-host dependencies. Some Qwen clinical responses came from a different A100 deployment, so its timing series is mixed.

The historical main environment used PyTorch 2.11 with CUDA 12.8 and Transformers 5.17.0. Install an appropriate PyTorch build explicitly, then use `requirements-gpu.txt`. SPARK requires its own environment using `requirements-spark.txt` (Transformers 4.57.1); do not combine those two files. Dependencies for offline analysis are ranges; the release check records the actual environment used to validate this publication.

Checkpoint IDs and revisions are in `data/config/models.json`. Model access can require acceptance of the provider's terms. The SPARK loader enables repository code for its pinned revision and applies a hash-checked activation-cast correction. Other models do not enable custom repository code. The local runner refuses unexpected precision, offload or prompt overflow.

The public runners are a portable implementation of the documented settings. Their GPU and API branches have not been rerun for publication; offline tests do not certify compatibility with every driver or future provider response.

## Credentials and output records

If deliberately making new requests, set the relevant environment variable: `GEMINI_API_KEY`, `OPENAI_API_KEY` or `OPENROUTER_API_KEY`. Never paste keys into a notebook or commit them. No credentials are included in this repository.

API calls are disabled in the default notebook flow. The helper writes request receipts under ignored `outputs/`, never authentication headers. A transport failure retains its pending marker and requires manual reconciliation before any retry. Estimated API cost is not an invoice. Raw responses, source excerpts, model caches, private review logs and operational deployment files are excluded from publication.
