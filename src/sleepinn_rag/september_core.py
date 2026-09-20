"""Auditable, model-independent operations for the September retrieval experiment.

No GPU imports at module load. Raw source spans and 1-based physical pages are
retained through packing; dataset knowledge-page indices are converted on input.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from difflib import SequenceMatcher
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import runpy


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_dictionaries(eval_dir):
    tree = ast.parse(
        (Path(eval_dir) / "sleepinn_rag/generation.py").read_text(encoding="utf-8")
    )
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "ABBREVIATIONS",
                    "CLINICAL_SYNONYMS",
                }:
                    result[target.id] = ast.literal_eval(node.value)
    assert len(result) == 2
    return result


def expand_abbreviations(text, mapping):
    for abbreviation, expansion in mapping.items():
        text = re.sub(
            rf"\b{re.escape(abbreviation)}\b",
            f"{abbreviation} ({expansion})",
            text,
            flags=re.I,
        )
    return text


def expand_terms(text, mapping):
    # Preserve the existing BM25 expansion rules, including their substring matching.
    terms = [
        term
        for key, values in mapping.items()
        if key in text.lower()
        for term in values[:2]
    ]
    return text + (" " + " ".join(terms) if terms else "")


def strategy_matrix(eval_dir):
    names = runpy.run_path(str(Path(eval_dir) / "sleepinn_rag/benchmark/presets.py"))[
        "PRESET_PAPER"
    ]
    excluded = {"dense_only_pageagg", "bm25_only_pageagg", "hybrid_rrf_pageagg"}
    names = ["dense_raw", "hybrid_rrf_no_terms", "dense_bm25_interleave"] + [
        n for n in names if n not in excluded
    ]
    matrix = []
    for name in names:
        both = any(part in name for part in ("hybrid", "concat", "interleave"))
        flags = {
            "strategy": name,
            "dense": both or "dense" in name,
            "bm25": both or "bm25" in name,
            "rrf": "rrf" in name,
            "rerank": "rerank" in name,
            "scope": name.startswith("scope_"),
            "mq": "_mq" in name,
            "jiwar": "_jiwar" in name,
            "pageagg": "_pageagg" in name,
            "abbreviations": name != "dense_raw",
            "bm25_terms": name != "hybrid_rrf_no_terms",
            "interleave": "interleave" in name,
        }
        matrix.append(flags)
    assert len(matrix) == 31 and len({r["strategy"] for r in matrix}) == 31
    return matrix


def deduplicate(indices):
    return list(dict.fromkeys(indices))


def fuse_lists(dense_lists, bm25_lists, flags, rrf_k=60):
    if flags["rrf"]:
        scores = defaultdict(float)
        for listing in dense_lists + bm25_lists:
            for rank, index in enumerate(listing, 1):
                scores[index] += 1.0 / (rrf_k + rank)
        order = sorted(scores, key=lambda idx: (-scores[idx], idx))
        return order, dict(scores)
    dense = deduplicate(i for listing in dense_lists for i in listing)
    bm25 = deduplicate(i for listing in bm25_lists for i in listing)
    if flags.get("interleave"):
        merged = []
        for rank in range(max(len(dense), len(bm25))):
            if rank < len(dense):
                merged.append(dense[rank])
            if rank < len(bm25):
                merged.append(bm25[rank])
        return deduplicate(merged), {}
    return deduplicate(dense + bm25), {}


def aggregate_pages(ranked_scores, chunks, limit=3):
    by_page = defaultdict(list)
    for index, score in ranked_scores:
        chunk = chunks[index]
        by_page[(chunk["source_pdf"], chunk["page"])].append(score)
    weights = [1.0, 0.5, 0.25]
    page_scores = {
        key: sum(
            weight * score
            for weight, score in zip(weights, sorted(values, reverse=True))
        )
        for key, values in by_page.items()
    }
    pages = sorted(page_scores, key=lambda p: (-page_scores[p], p))[:limit]
    selected = [
        index
        for index, _ in ranked_scores
        if (chunks[index]["source_pdf"], chunks[index]["page"]) in pages
    ]
    return selected, [
        {"source_pdf": key[0], "page": key[1], "score": page_scores[key]}
        for key in pages
    ]


def merge_spans(indices, chunks, pages):
    """Merge overlapping source offsets, so overlap text is not counted twice."""
    grouped = defaultdict(list)
    for index in sorted(indices):
        chunk = chunks[index]
        grouped[(chunk["source_pdf"], chunk["page"])].append(chunk)
    spans = []
    for (source, page), records in grouped.items():
        ranges = []
        for chunk in sorted(records, key=lambda c: (c["start"], c["end"])):
            if ranges and chunk["start"] <= ranges[-1]["end"]:
                ranges[-1]["end"] = max(ranges[-1]["end"], chunk["end"])
                ranges[-1]["chunk_indices"].append(chunk["index"])
            else:
                ranges.append(
                    {
                        "start": chunk["start"],
                        "end": chunk["end"],
                        "chunk_indices": [chunk["index"]],
                    }
                )
        for record in ranges:
            spans.append(
                {
                    "source_pdf": source,
                    "page": page,
                    **record,
                    "text": pages[(source, page)][record["start"] : record["end"]],
                }
            )
    return spans


def make_blocks(anchors, chunks, pages, jiwar=False, max_constituents=8):
    blocks, used = [], set()
    for rank, anchor in enumerate(anchors[:8]):
        if anchor in used:
            continue
        if len(used) >= max_constituents:
            break
        selected = [anchor]
        if jiwar and rank < 2:
            for neighbor in (anchor - 1, anchor + 1):
                if not 0 <= neighbor < len(chunks):
                    continue
                a, n = chunks[anchor], chunks[neighbor]
                if (
                    n["source_pdf"] == a["source_pdf"]
                    and abs(n["page"] - a["page"]) <= 1
                ):
                    if (
                        neighbor not in used
                        and len(used) + len(selected) < max_constituents
                    ):
                        selected.append(neighbor)
        used.update(selected)
        blocks.append(
            {
                "anchor_index": anchor,
                "chunk_indices": sorted(selected),
                "spans": merge_spans(selected, chunks, pages),
            }
        )
    return blocks


def render_blocks(blocks):
    return "\n\n".join(
        "\n".join(
            f"[{s['source_pdf']} | PDF page {s['page']}]\n{s['text']}"
            for s in block["spans"]
        )
        for block in blocks
    )


def apply_budget(blocks, tokenizer, budget=4096):
    """Limit the actual rendered context; retain provenance only for delivered text."""

    def count(value):
        return len(tokenizer.encode(render_blocks(value), add_special_tokens=False))

    if count(blocks) <= budget:
        return blocks, count(blocks)
    accepted = []
    for block in blocks:
        current = {
            "anchor_index": block["anchor_index"],
            "chunk_indices": block["chunk_indices"],
            "spans": [],
        }
        for span in block["spans"]:
            proposal = {**current, "spans": current["spans"] + [span]}
            if count(accepted + [proposal]) <= budget:
                current = proposal
                continue
            low, high = 0, len(span["text"])
            while low < high:
                middle = (low + high + 1) // 2
                clipped = {
                    **span,
                    "text": span["text"][:middle],
                    "end": span["start"] + middle,
                }
                candidate = {**current, "spans": current["spans"] + [clipped]}
                if count(accepted + [candidate]) <= budget:
                    low = middle
                else:
                    high = middle - 1
            if low:
                clipped = {
                    **span,
                    "text": span["text"][:low],
                    "end": span["start"] + low,
                }
                current["spans"].append(clipped)
            if current["spans"]:
                accepted.append(current)
            assert count(accepted) <= budget
            return accepted, count(accepted)
        if current["spans"]:
            accepted.append(current)
    assert count(accepted) <= budget
    return accepted, count(accepted)


def normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


@lru_cache(maxsize=100_000)
def evidence_match(unit, text):
    # Exact comparison ignores OCR whitespace/hyphenation, without changing letters or numbers.
    compact_unit = re.sub(r"[^a-z0-9]", "", unit.lower())
    compact_text = re.sub(r"[^a-z0-9]", "", text.lower())
    if compact_unit and compact_unit in compact_text:
        return True
    unit, text = normalize(unit), normalize(text)
    if not unit or not text:
        return False
    if unit in text:
        return True
    if len(unit) < 25:
        return False
    if len(text) < len(unit):
        return SequenceMatcher(None, unit, text).ratio() >= 0.82
    for start in range(0, max(1, len(text) - len(unit) + 1), max(20, len(unit) // 4)):
        if (
            SequenceMatcher(None, unit, text[start : start + len(unit) + 40]).ratio()
            >= 0.82
        ):
            return True
    return False


def gold_pages(item, unit_index):
    aligned = item.get("metadata", {}).get("evidence_pages")
    if aligned and len(aligned) == len(item["gold_evidence_units"]):
        return {int(aligned[unit_index]) + 1}
    return set(range(int(item["page_start"]) + 1, int(item["page_end"]) + 2))


def score_blocks(item, blocks, prefix=""):
    hits = []
    for block in blocks:
        found = []
        for i, unit in enumerate(item["gold_evidence_units"]):
            valid = [
                s["text"]
                for s in block["spans"]
                if s["source_pdf"] == item["source_pdf"]
                and s["page"] in gold_pages(item, i)
            ]
            found.append(
                any(evidence_match(unit, t) for t in valid)
                or (len(valid) > 1 and evidence_match(unit, "\n".join(valid)))
            )
        hits.append(found)
    total = len(item["gold_evidence_units"])
    if not total:
        raise ValueError("Knowledge item has no gold evidence")
    scores = {}
    for k in (1, 3, 5, 8):
        scores[f"{prefix}evidence_recall@{k}"] = (
            sum(any(row[i] for row in hits[:k]) for i in range(total)) / total
        )
        scores[f"{prefix}page_hit@{k}"] = int(
            any(
                s["source_pdf"] == item["source_pdf"]
                and int(item["page_start"]) + 1
                <= s["page"]
                <= int(item["page_end"]) + 1
                for block in blocks[:k]
                for s in block["spans"]
            )
        )
    scores[f"{prefix}evidence_mrr"] = next(
        (1 / (i + 1) for i, row in enumerate(hits) if any(row)), 0.0
    )
    return scores


def parse_queries(response, original):
    """Extract three distinct query lines; accept JSON and trim surplus query lines."""
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned).strip()
    if cleaned.startswith("["):
        try:
            values = json.loads(cleaned)
            if isinstance(values, list) and all(isinstance(v, str) for v in values):
                response = "\n".join(values)
        except json.JSONDecodeError:
            pass
    lines = []
    for line in response.splitlines():
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip(' \t"*')
        if not line or line.startswith("```"):
            continue
        if line.lower().startswith(
            ("here are", "search queries:", "queries:", "alternative queries:")
        ):
            continue
        if line.startswith("#") or (line.endswith(":") and "quer" in line.lower()):
            continue
        if line.lower() == original.strip().lower():
            continue
        if 5 <= len(line) <= 800 and line.lower() not in {q.lower() for q in lines}:
            lines.append(line)
    if len(lines) < 3:
        raise ValueError(
            f"Expected at least three distinct query lines, received {len(lines)}"
        )
    return lines[:3]


def trim_generation_padding(token_ids, pad_token_id):
    """Remove only trailing batch padding; preserve the model's turn terminator."""
    token_ids = list(token_ids)
    end = len(token_ids)
    while end and token_ids[end - 1] == pad_token_id:
        end -= 1
    return token_ids[:end]
