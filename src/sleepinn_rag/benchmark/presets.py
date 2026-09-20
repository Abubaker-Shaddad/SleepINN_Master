"""Strategy presets for benchmarking."""

# Core strategy groups
STRAT_BASELINES = [
    "dense_only",
    "bm25_only",
    "dense_bm25_concat",
    "hybrid_rrf",
]

STRAT_RERANK = [
    "hybrid_rrf_rerank",
    "dense_rerank",
    "bm25_rerank",
]

STRAT_MULTI_QUERY = [
    "dense_only_mq",
    "bm25_only_mq",
    "hybrid_rrf_mq",
    "hybrid_rrf_rerank_mq",
]

STRAT_JIWAR = [
    "dense_only_jiwar",
    "bm25_only_jiwar",
    "hybrid_rrf_jiwar",
    "hybrid_rrf_rerank_jiwar",
    "hybrid_rrf_rerank_mq_jiwar",
    "scope_hybrid_rrf_rerank_jiwar",
    "scope_hybrid_rrf_rerank_mq_jiwar",
]

STRAT_PAGE_AGG = [
    "dense_only_pageagg",
    "bm25_only_pageagg",
    "hybrid_rrf_pageagg",
    "hybrid_rrf_rerank_pageagg",
    "hybrid_rrf_rerank_mq_pageagg",
    "scope_hybrid_rrf_rerank_mq_pageagg",
]

STRAT_PAGE_AGG_JIWAR = [
    "hybrid_rrf_rerank_pageagg_jiwar",
    "hybrid_rrf_rerank_mq_pageagg_jiwar",
    "scope_hybrid_rrf_rerank_pageagg_jiwar",
    "scope_hybrid_rrf_rerank_mq_pageagg_jiwar",
]

STRAT_SCOPE = [
    "scope_dense_only",
    "scope_bm25_only",
    "scope_hybrid_rrf",
    "scope_hybrid_rrf_rerank",
    "scope_hybrid_rrf_rerank_mq",
]

# Optional strategy groups
STRAT_MQV2 = ["hybrid_rrf_rerank_mqv2"]

STRAT_SED = [
    "hybrid_rrf_rerank_sed",
    "hybrid_rrf_rerank_mq_sed",
    "hybrid_rrf_rerank_mqv2_sed",
]

STRAT_IQRAA = [
    "hybrid_rrf_rerank_iqraa_boost",
    "hybrid_rrf_rerank_iqraa_restrict",
    "hybrid_rrf_rerank_mqv2_iqraa_boost",
    "hybrid_rrf_rerank_mqv2_sed_iqraa_boost",
]

STRAT_QWEN_RERANKER = [
    "hybrid_rrf_qwen_rerank",
    "hybrid_rrf_qwen_rerank_mq",
    "hybrid_rrf_qwen_rerank_mqv2",
    "hybrid_rrf_qwen_rerank_sed",
    "hybrid_rrf_qwen_rerank_iqraa_boost",
    "hybrid_rrf_qwen_rerank_mqv2_sed_iqraa_boost",
]

# Presets
PRESET_QUICK = STRAT_BASELINES + ["hybrid_rrf_rerank"]

PRESET_CORE = list(
    dict.fromkeys(
        STRAT_BASELINES
        + STRAT_RERANK
        + STRAT_MULTI_QUERY
        + STRAT_PAGE_AGG
        + STRAT_JIWAR
        + STRAT_PAGE_AGG_JIWAR
        + STRAT_SCOPE
    )
)

PRESET_OPTIONAL = list(
    dict.fromkeys(STRAT_MQV2 + STRAT_SED + STRAT_IQRAA + STRAT_QWEN_RERANKER)
)

PRESET_BEST_OF_EACH = [
    "hybrid_rrf_rerank",
    "hybrid_rrf_rerank_mq",
    "hybrid_rrf_rerank_mq_pageagg",
    "hybrid_rrf_rerank_mq_jiwar",
    "hybrid_rrf_rerank_mq_pageagg_jiwar",
    "scope_hybrid_rrf_rerank_mq",
    "scope_hybrid_rrf_rerank_mq_pageagg_jiwar",
]

PRESET_BEST_WITH_OPTIONAL = [
    "hybrid_rrf_rerank",
    "hybrid_rrf_rerank_mq",
    "hybrid_rrf_rerank_mqv2_sed",
    "hybrid_rrf_rerank_mqv2_sed_iqraa_boost",
    "hybrid_rrf_rerank_mq_pageagg",
    "hybrid_rrf_rerank_mq_jiwar",
    "hybrid_rrf_rerank_mq_pageagg_jiwar",
    "scope_hybrid_rrf_rerank_mq",
    "scope_hybrid_rrf_rerank_mq_pageagg_jiwar",
]

PRESET_ALL = list(dict.fromkeys(PRESET_CORE + PRESET_OPTIONAL))

# PAPER preset (systematic ablation)
# Group 1: Baselines (no reranking, no MQ)
# Group 2: + Reranking
# Group 3: + Multi-Query
# Group 4: + SCOPE-RAG-lite
# Group 5: + Jiwar (adjacent-chunk packing)
# Group 6: + Page Aggregation (± Jiwar)

PAPER_G1_BASELINES = [
    "dense_only",
    "bm25_only",
    "dense_bm25_concat",
    "hybrid_rrf",
]

PAPER_G2_RERANK = [
    "hybrid_rrf_rerank",
    "dense_rerank",
    "bm25_rerank",
]

PAPER_G3_MULTI_QUERY = [
    "dense_only_mq",
    "bm25_only_mq",
    "hybrid_rrf_mq",
    "hybrid_rrf_rerank_mq",
]

PAPER_G4_SCOPE = [
    "scope_dense_only",
    "scope_bm25_only",
    "scope_hybrid_rrf",
    "scope_hybrid_rrf_rerank",
    "scope_hybrid_rrf_rerank_mq",
]

PAPER_G5_JIWAR = [
    "dense_only_jiwar",
    "bm25_only_jiwar",
    "hybrid_rrf_jiwar",
    "hybrid_rrf_rerank_jiwar",
    "hybrid_rrf_rerank_mq_jiwar",
    "scope_hybrid_rrf_rerank_jiwar",
    "scope_hybrid_rrf_rerank_mq_jiwar",
]

PAPER_G6_PAGE_AGG = [
    "dense_only_pageagg",
    "bm25_only_pageagg",
    "hybrid_rrf_pageagg",
    "hybrid_rrf_rerank_pageagg",
    "hybrid_rrf_rerank_mq_pageagg",
    "hybrid_rrf_rerank_pageagg_jiwar",
    "scope_hybrid_rrf_rerank_pageagg_jiwar",
    "scope_hybrid_rrf_rerank_mq_pageagg_jiwar",
]

PRESET_PAPER = list(
    dict.fromkeys(
        PAPER_G1_BASELINES
        + PAPER_G2_RERANK
        + PAPER_G3_MULTI_QUERY
        + PAPER_G4_SCOPE
        + PAPER_G5_JIWAR
        + PAPER_G6_PAGE_AGG
    )
)
