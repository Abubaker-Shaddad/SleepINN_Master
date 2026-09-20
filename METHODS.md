# Methods and interpretation

## Databanks

The knowledge set contains 1,000 ICSD-3-TR questions and 215 AASM Scoring Manual Version 3 questions. Its three endpoints are multiple-choice correctness, free-text reference agreement and criteria coverage. The initial set contained 699 MCQs, 403 free-text questions and 113 criteria questions; AI revision changed three question types, giving 702, 400 and 113 in the final bank. The source records document revisions to 134 ICSD items, including changes to evidence or metadata that are not all visible in the public projection.

Clinical questions are adapted from six published casebooks, with 20 cases per book. Tasks include diagnosis, suspected diagnosis, clinical interpretation and diagnostic evaluation. Knowledge source-page fields are zero-based PDF indices; clinical page fields are one-based physical PDF pages. Source names and locations are retained, while verbatim evidence excerpts are omitted from the public files.

The author reports completed expert review of both databanks. The release preserves that statement without inventing historical acceptance logs. New reviews record the original item, changes, reviewer identifier, decision, source check and timestamp. Every question must be decided before final export. The two completed exports can then be combined with `assemble_final`.

## Retrieval

The implementation compares 31 combinations of dense retrieval, BM25, concatenation, balanced alternating merge, reciprocal rank fusion, structural context, reranking, multiple-query expansion, page aggregation and neighbouring passages. BGE-M3 provides dense embeddings; the cross-encoder is `cross-encoder/ms-marco-MiniLM-L12-v2`. Reranking is used only in the configurations whose flags enable it.

RRF uses one-based ranks and a constant of 60. Chunks contain up to 1,024 characters with 512-character overlap. Fifty candidates per retriever/query feed the subsequent stages. SCOPE adds document and section context. Jiwar extends the top two anchors with neighbouring passages, then merges overlapping source spans. Delivered contexts are limited to eight blocks and 4,096 tokens under a common reference tokenizer.

The selected `scope_hybrid_rrf_rerank_jiwar` configuration does not use multiple-query expansion. Multi-query configurations use three Gemma 4 12B query variants. The original question alone is used to query evidence; reference answers are used for retrieval evaluation, not ordinary retrieval. Perfect retrieval is a separate oracle condition based on known source evidence, not a deployable search method.

All configurations were evaluated on the same 1,215 questions. This permits paired method comparisons but does not provide an independent held-out estimate of the selected pipeline's advantage. The timing columns estimate amortized component cost; they are not isolated request latency or total answer-generation time.

## Models

Eight local models are included: Llama 3.2 1B, Llama 3.1 8B, SPARK X2.5 4B, Gemma 3 12B, Gemma 4 12B, Phi-4 14B, Gemma 3 27B and Qwen 3.8 27B. Knowledge generation used NF4; clinical generation used NF4 and BF16. Gemini 3.8 Flash, GPT 5.5 and Claude Opus 4.8 add three closed clinical systems. The resulting clinical cohort contains 19 systems, not 20.

Local generation uses pinned revisions, greedy decoding, a batch size of one and a 1,024-token output limit. NF4 uses double quantization with BF16 computation. BF16 is finite precision, not exact arithmetic. SPARK uses a separate Transformers environment and the documented activation-cast correction in its loader. Closed APIs expose different generation controls and do not disclose comparable model precision. See `data/config/models.json` and the execution settings.

## Knowledge assessment

- **Multiple choice:** compare the selected option with the reference key. Preserve ambiguous output formats for adjudication instead of treating every initial “A” as option A.
- **Free text:** accept semantically equivalent answers using the saved rubric. The primary score is strict correct=1, partial/incorrect=0; partial credit is secondary.
- **Criteria:** average coverage of the fixed reference components: covered=1, partial=0.5, missing/contradicted=0. Ungradable components make that item missing.

The original no-RAG and selected-RAG semantic assessments use Gemini 3.8 Flash. Perfect-retrieval semantic assessments use GLM 5.3 Flash through OpenRouter. Original scores were retained unchanged. Differences between perfect and selected retrieval can therefore reflect both context and judge. No claim is made that the judges are interchangeable. Saved question-level scores, validity flags, reasons and coverage support reanalysis without another API call.

## Clinical assessment and statistics

Gemini 3.8 Flash and GLM 5.3 Flash independently assessed blinded responses using the same rubric. Model name, precision and retrieval condition are not supplied to the judge. Historical prompt wording is retained as a record of the scoring protocol; subsequent expert review of the databank does not retroactively turn automated answer labels into physician scores.

Exact consensus requires the same valid label—correct, partial or incorrect—from both judges. Both the no-RAG and RAG response must meet this rule for a case pair to enter the main RAG comparison. The final cohort contains 4,053 response-level agreements and 1,831 complete agreed pairs out of 2,280 possible pairs. Pair retention varies by system. Invalid and ungradable scores remain missing. Single-judge sensitivity results and full-population bounds are supplied alongside the agreement-only results.

Per-system contrasts use paired differences, source-book-stratified bootstrap intervals and exact McNemar tests for binary endpoints. Aggregate contrasts give equal weight to each fixed system (or matched model/condition for precision). A case-cluster bootstrap samples whole cases within each of the six books, keeping responses from the same case together across models. The implementation uses 30,000 resamples and 100,000 shared-case sign permutations, with seed 20260919. Permutation inference assumes sign symmetry under the null. P values are exploratory and unadjusted; confidence intervals are not proof of equivalence.

The current 19-system aggregate is 68.82% without RAG and 73.67% with RAG: +4.85 percentage points. Earlier 20-system summaries are not the primary cohort. The saved aggregate estimates are in `results/analysis/clinical/AGGREGATE_RESULTS.json` and are reproduced by notebook 09.

Model-size associations use the eight local parameter counts. Closed-model parameter counts are unknown and are not treated as measured values. Family, architecture, training and size are confounded in this small model sample. GPU measurements report allocated and reserved memory separately; hardware, prompt length and deployment differences limit latency comparisons.
