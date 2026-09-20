"""Offline integrity checks for the published study files."""
from collections import Counter
import pandas as pd
from .databank import validate_items
from .io import project_root, read_json, read_jsonl, file_hash, digest, answer_frame
from .statistics import consensus_scores


def verify_release():
    root = project_root()
    bank = read_jsonl(root / 'data/final/knowledge.jsonl') + read_jsonl(root / 'data/final/clinical.jsonl')
    counts = validate_items(bank)
    if counts['items'] != 1335:
        raise ValueError('Unexpected databank population')
    for domain, expected, per_system in [('knowledge', 29160, 3645), ('clinical', 4560, 240)]:
        frame = answer_frame(domain)
        if len(frame) != expected or not frame.groupby(['model_id','precision']).size().eq(per_system).all():
            raise ValueError('Incomplete answer population')
        if not frame.apply(lambda r: digest(r.answer.encode()) == r.answer_sha256, axis=1).all():
            raise ValueError('Answer text hash mismatch')
    comparison = pd.read_csv(root / 'results/analysis/clinical/judge_comparison_19_systems.csv')
    scored = consensus_scores(comparison)
    if len(scored) != 4560 or int(scored.included_exact_consensus.sum()) != 4053:
        raise ValueError('Clinical judge population differs')
    pairs = scored.pivot(index=['model_id','precision','item_id'], columns='mode', values='strict_score')
    if len(pairs[['no_rag','with_rag']].dropna()) != 1831:
        raise ValueError('Clinical paired inclusion differs')
    manifest = read_json(root / 'data/FILE_SHA256.json')
    for name, expected in manifest.items():
        if file_hash(root / name) != expected:
            raise ValueError('Release file differs: ' + name)
    return dict(questions=1335, knowledge_answers=29160, clinical_answers=4560,
                clinical_systems=19, consensus_pairs=1831, verified_files=len(manifest))


if __name__ == '__main__':
    print(verify_release())
