"""Paired estimates with visible denominators and missing-data bounds."""
import numpy as np
import pandas as pd
from scipy.stats import binomtest


def paired_effect(
    frame, value, strata=None, planned_pairs=None, seed=20260919, resamples=10000
):
    """Bootstrap cases within source books. Invalid scores remain missing."""
    keys = ["item_id", "mode"]
    if frame.duplicated(keys).any():
        raise ValueError("Select one model, precision, and endpoint before pairing")
    population = frame.item_id.nunique() if planned_pairs is None else planned_pairs
    pairs = frame.pivot(index="item_id", columns="mode", values=value)
    pairs = pairs.reindex(columns=["no_rag", "with_rag"]).dropna()
    if pairs.empty:
        raise ValueError("No complete pairs")
    differences = (pairs.with_rag - pairs.no_rag).to_numpy(float)
    rng = np.random.default_rng(seed)
    if strata:
        lookup = frame.drop_duplicates("item_id").set_index("item_id")[strata]
        labels = lookup.reindex(pairs.index).fillna("unknown").to_numpy()
    else:
        labels = np.zeros(len(pairs))
    sums = np.zeros(resamples)
    for label in pd.unique(labels):
        block = differences[labels == label]
        sums += rng.choice(block, size=(resamples, len(block)), replace=True).sum(
            axis=1
        )
    lower, upper = np.quantile(sums / len(pairs) * 100, [0.025, 0.975])
    binary = bool(np.isin(pairs.to_numpy(), [0, 1]).all())
    gained = (
        int(((pairs.no_rag == 0) & (pairs.with_rag == 1)).sum()) if binary else None
    )
    lost = int(((pairs.no_rag == 1) & (pairs.with_rag == 0)).sum()) if binary else None
    p = (
        binomtest(gained, gained + lost, 0.5).pvalue
        if binary and gained + lost
        else (1.0 if binary else None)
    )
    missing = population - len(pairs)
    if missing < 0:
        raise ValueError("Planned population is smaller than the observed sample")
    return dict(
        planned_pairs=population,
        complete_pairs=len(pairs),
        excluded_pairs=missing,
        no_rag_pct=pairs.no_rag.mean() * 100,
        with_rag_pct=pairs.with_rag.mean() * 100,
        delta_pp=differences.mean() * 100,
        ci95_low_pp=lower,
        ci95_high_pp=upper,
        gained=gained,
        lost=lost,
        exact_mcnemar_p_unadjusted=p,
        full_case_lower_delta_pp=(differences.sum() - missing) / population * 100,
        full_case_upper_delta_pp=(differences.sum() + missing) / population * 100,
    )


def consensus_scores(comparison):
    """Require exact correct/partial/incorrect agreement for each answer."""
    frame = comparison.copy()
    valid = {"correct", "partial", "incorrect"}
    eligible = frame.gemini_label.isin(valid) & frame.glm_label.isin(valid)
    agreed = eligible & frame.gemini_label.eq(frame.glm_label)
    frame["included_exact_consensus"] = agreed
    frame["strict_score"] = frame.gemini_label.map(
        {"correct": 1.0, "partial": 0.0, "incorrect": 0.0}
    ).where(agreed)
    frame["partial_credit"] = frame.gemini_label.map(
        {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
    ).where(agreed)
    return frame


def clinical_rag_table(comparison, resamples=10000):
    frame = consensus_scores(comparison)
    rows = []
    for (model, precision), group in frame.groupby(
        ["model_id", "precision"], sort=True
    ):
        estimate = paired_effect(
            group,
            "strict_score",
            strata="source_book",
            planned_pairs=120,
            resamples=resamples,
        )
        rows.append(dict(model_id=model, precision=precision, **estimate))
    return pd.DataFrame(rows)
