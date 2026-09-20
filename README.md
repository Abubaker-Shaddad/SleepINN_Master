# Large language models and retrieval in sleep medicine

Code, databanks and results accompanying the master's thesis *Towards Integration of Trustworthy Large Language Models into Sleep Diagnostics*.

**[Browse the databank](https://abubaker-shaddad.github.io/SleepINN_Master/results/dataset_viewer.html)** · **[Browse model answers](https://abubaker-shaddad.github.io/SleepINN_Master/results/model_results_viewer.html)** · **[Start with the notebooks](notebooks/00_start_here.ipynb)**

The HTML files also work locally: download the repository and open **`results/dataset_viewer.html`** or **`results/model_results_viewer.html`** in a modern browser. No API key or Python installation is needed to browse them.

## What is included

| Folder | Contents |
|---|---|
| [`notebooks/`](notebooks) | Twelve numbered walkthroughs: databank creation, human review, retrieval, model execution and analysis |
| [`src/`](src) | The retrieval algorithms, correction interface, model runners, scoring rules and statistical functions |
| [`data/`](data) | Candidate, AI-revised and final databanks; model settings, rubrics and prompt hashes |
| [`results/`](results) | Model answers, question-level scores, retrieval measurements, summary tables and HTML viewers |
| [`tests/`](tests) | Offline checks for review exports, retrieval fusion, blinding, prompt reconstruction and release integrity |

## Study population

- **Knowledge:** 1,215 questions: 702 multiple-choice, 400 free-text and 113 criteria questions. Eight local NF4 models answered without RAG, with SCOPE–Jiwar and with perfect retrieval: **29,160 answers**.
- **Clinical:** 120 cases. Eight local models at NF4 and BF16 plus three closed models make **19 model–precision systems**, each evaluated with and without RAG: **4,560 answers**.
- **Retrieval:** 31 configurations × 1,215 knowledge questions: **37,665 comparisons**.

The unmatched Gemma 4 26B system is not part of this thesis cohort. Historical answer text and recorded scores are preserved; this publication does not regenerate or reassess answers.

## Review and final databank

The workflow is **source-based drafting → AI revision where applicable → human review → final databank**. Knowledge and clinical questions have separate human-review forms with **accept, reject and correct** controls. Final export requires a decision for every item and writes a completion manifest with file hashes.

The thesis author reports that a sleep expert completed review of both supplied databanks without further changes to their released content. The repository records this at release level in [`data/REVIEW_STATUS.json`](data/REVIEW_STATUS.json); it does not create retrospective item-by-item review logs. The forms in notebooks 02 and 03 are available for recording a new review.

Expert review of the **dataset** and automated assessment of **model responses** are distinct. Clinical answer scoring uses the saved independent Gemini and GLM labels. See [Methods](METHODS.md) for the agreement rule, missing scores and interpretation limits.

## Run the offline notebooks

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python -m pip install -e .
python -m unittest discover -s tests
python -m jupyterlab notebooks
```

Start with `00_start_here.ipynb`. All default notebook cells run without model downloads, GPU work or API calls. Fresh experiments are opt-in and go under ignored `outputs/`.

For GPU execution, source preparation, environment differences and known reproduction limits, read [REPRODUCING.md](REPRODUCING.md). Public data omit licensed book excerpts and full retrieved contexts, so rerunning source-dependent generation requires your own source material. The original model-answer text and all analysis inputs remain available.

## Useful entry points

- Human review: [`review.py`](src/sleepinn_study/review.py)
- Final assembly and paired prompts: [`workflow.py`](src/sleepinn_study/workflow.py)
- Dense/BM25 retrieval and reranking: [`retrieval_experiment.py`](src/sleepinn_study/retrieval_experiment.py)
- RRF, balanced merge, neighbouring passages and context packing: [`september_core.py`](src/sleepinn_rag/september_core.py)
- Structural context: [`scope.py`](src/sleepinn_rag/kb/scope.py)
- Local inference: [`benchmark.py`](src/sleepinn_study/benchmark.py)
- Clinical paired statistics: [`statistics.py`](src/sleepinn_study/statistics.py) and [`aggregate.py`](src/sleepinn_study/aggregate.py)
- Public-file hashes: [`data/FILE_SHA256.json`](data/FILE_SHA256.json)

This repository contains research material, not a validated clinical application. The source manuals and casebooks retain their respective rights; see [DATA_NOTICE.md](DATA_NOTICE.md).
