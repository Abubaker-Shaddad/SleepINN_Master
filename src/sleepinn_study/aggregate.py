"""Clinical contrasts with shared cases kept together across models."""
import numpy as np
import pandas as pd
from .statistics import consensus_scores


def aggregate_clinical(comparison, *, resamples=30000, permutations=100000, seed=20260919):
    """Reproduce the thesis's five fixed-cohort aggregate contrasts.

    Bootstrap cases within books. For the exploratory p value, flip a shared
    sign per case under the assumption of sign symmetry. Do not treat each
    model answer as an independent patient.
    """
    d = consensus_scores(comparison).rename(columns={'strict_score': 'score'})
    cases = sorted(d.item_id.unique())
    if d.groupby('item_id').source_book.nunique().max() != 1:
        raise ValueError('Each case must belong to one source book')
    books = d.groupby('item_id').source_book.first().reindex(cases)
    rng = np.random.default_rng(seed)
    counts = np.zeros((resamples, len(cases)), dtype=np.int16)
    for book in sorted(books.unique()):
        ix = np.flatnonzero(books.to_numpy() == book)
        counts[:, ix] = rng.multinomial(len(ix), np.full(len(ix), 1 / len(ix)), size=resamples)

    def estimate(frame, keys, contrast, before, after, label):
        wide = frame.pivot(index=['item_id'] + keys, columns=contrast, values='score')[[before, after]].dropna()
        groups = sorted(set(tuple(i[1:]) for i in wide.index))
        a = np.zeros((len(cases), len(groups)))
        b = np.zeros_like(a)
        eligible = np.zeros_like(a)
        for j, group in enumerate(groups):
            sub = wide.xs(group if len(group) > 1 else group[0], level=keys)
            loc = pd.Index(cases).get_indexer(sub.index)
            a[loc, j], b[loc, j], eligible[loc, j] = sub[before], sub[after], 1
        n = eligible.sum(axis=0)
        den = counts @ eligible
        if not len(groups) or np.any(n == 0) or np.any(den == 0):
            raise ValueError('Insufficient paired coverage for this aggregate')
        delta = 100 * ((b - a).sum(axis=0) / n).mean()
        boot = 100 * ((counts @ (b - a)) / den).mean(axis=1)
        contributions = 100 * ((b - a) / n).mean(axis=1)
        exceed = 0
        for start in range(0, permutations, 5000):
            size = min(5000, permutations - start)
            signs = rng.integers(0, 2, size=(size, len(cases)), dtype=np.int8) * 2 - 1
            exceed += np.count_nonzero(np.abs(signs @ contributions) >= abs(delta) - 1e-12)
        return dict(label=label, groups=len(groups), pairs=int(n.sum()),
                    before_pct=100 * (a.sum(axis=0) / n).mean(), after_pct=100 * (b.sum(axis=0) / n).mean(),
                    delta_pp=delta, ci95=np.quantile(boot, [.025, .975]).tolist(),
                    p_unadjusted=(exceed + 1) / (permutations + 1))

    local = d[d.precision.isin(['NF4', 'BF16'])]
    matched = set(local[local.precision.eq('NF4')].model_id) & set(local[local.precision.eq('BF16')].model_id)
    precision = local[local.model_id.isin(matched)]
    return [
        estimate(d, ['model_id', 'precision'], 'mode', 'no_rag', 'with_rag', 'rag_all_19_systems'),
        estimate(local, ['model_id', 'precision'], 'mode', 'no_rag', 'with_rag', 'rag_local_16_systems'),
        estimate(precision, ['model_id', 'mode'], 'precision', 'NF4', 'BF16', 'precision_all_16_conditions'),
        estimate(precision[precision['mode'].eq('no_rag')], ['model_id', 'mode'], 'precision', 'NF4', 'BF16', 'precision_no_rag'),
        estimate(precision[precision['mode'].eq('with_rag')], ['model_id', 'mode'], 'precision', 'NF4', 'BF16', 'precision_with_rag'),
    ]
