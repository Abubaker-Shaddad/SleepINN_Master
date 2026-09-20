"""Rebuild the study's clinical records from the saved source-based drafts."""

from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from .databank import validate_items
from .io import file_hash, read_json


def rebuild_clinical_cases(authoring_folder):
    """Package the saved drafts; this does not write new vignettes or call an API.

    Input hashes bind the drafts to their original evidence audit and provenance.
    The audit records earlier source checks; this function does not recheck PDFs.
    """
    folder = Path(authoring_folder)
    provenance = read_json(folder / "PROVENANCE.json")
    for name, expected in provenance["input_sha256"].items():
        if file_hash(folder / name) != expected:
            raise ValueError(f"Clinical authoring input changed: {name}")

    sources = read_json(folder / "source_manifest.json")
    config = read_json(folder / "export_settings.json")
    evidence_by_case = defaultdict(list)
    for entry in read_json(folder / "evidence_audit.json"):
        if not entry["match"]:
            raise ValueError("The saved evidence audit contains an unmatched excerpt")
        evidence_by_case[(entry["book"], entry["number"])].append(entry)

    cases = []
    for source in sources:
        book_id = source["book_id"]
        drafts = read_json(folder / f"authored_book_{book_id}.json")
        if len(drafts) != 20 or len({draft["case"] for draft in drafts}) != 20:
            raise ValueError(
                f"Expected 20 distinct source-case anchors in book {book_id}"
            )
        for number, draft in enumerate(drafts, start=1):
            evidence = evidence_by_case[(book_id, number)]
            # Some draft excerpts carry a third field marking a visual source check.
            draft_quotes = [excerpt[:2] for excerpt in draft["evidence"]]
            if [[entry["page"], entry["quote"]] for entry in evidence] != draft_quotes:
                raise ValueError(f"Evidence does not match draft {book_id}/{number}")
            cases.append(_case_record(draft, source, number, evidence, config))

    validate_items(cases)
    if len(cases) != 120:
        raise ValueError("The study export must contain 120 clinical cases")
    return cases


def _case_record(draft, source, number, evidence, config):
    start, end = draft["pages"]
    if not 1 <= start <= end <= source["pages"]:
        raise ValueError("Clinical page range lies outside the source book")
    if not evidence or any(not start <= entry["page"] <= end for entry in evidence):
        raise ValueError("Missing evidence or evidence outside the case page range")
    if draft["type"] not in config["question_tasks"]:
        raise ValueError("Unknown clinical question task")
    if not draft["question"].endswith("?") or len(draft["question"].split()) < 25:
        raise ValueError("Missing or unusually short clinical vignette")
    if not draft["answer"] or not draft["rationale"]:
        raise ValueError("Missing clinical reference answer or rationale")

    flags = [
        "Invented patient findings, measurements or source citations",
        "Treating a source-based candidate answer as independently physician-validated",
    ]
    if draft["type"] == "suspected_diagnosis":
        flags.append(
            "Claiming a definitive diagnosis or unreported confirmatory test result"
        )
    if draft["type"] == "clinical_interpretation":
        flags.append(
            "Replacing the requested finding or interpretation with an unsupported diagnosis"
        )

    # Retain the original provenance fields when reproducing the study records.
    original = config["metadata"]
    metadata = {
        "benchmark_split": original["benchmark_split"],
        "dataset_version": original["dataset_version"],
        "author": original["author"],
        "authored_date": original["authored_date"],
        "generation_method": original["generation_method"],
        "api_generation_model": original["api_generation_model"],
        "api_sampling_parameters": original["api_sampling_parameters"],
        "source_book_id": source["book_id"],
        "source_case_id": draft["case"],
        "source_pdf_sha256": source["sha256"],
        "source_kind": draft.get("source_kind", "published_patient_case"),
        "page_numbering": original["page_numbering"],
        "question_task": draft["type"],
        "answer_rationale": draft["rationale"],
        "evidence_excerpts": [
            {
                "pdf_page": entry["page"],
                "text": entry["quote"],
                "normalized_match_extractor": entry["match"],
            }
            for entry in evidence
        ],
        "author_self_review": original["author_self_review"],
        "physician_review_status": original["physician_review_status"],
        "clinical_validity": original["clinical_validity"],
        "difficulty_status": original["difficulty_status"],
        "source_caveat": draft.get("note"),
    }
    if draft.get("visual_pages"):
        metadata["source_pages_visually_inspected_during_authoring"] = deepcopy(
            draft["visual_pages"]
        )
    if draft.get("review_pass_02"):
        metadata["additional_author_review"] = deepcopy(draft["review_pass_02"])

    return {
        "item_id": f"{config['item_id_prefix']}{source['book_id']:02d}_{number:03d}",
        "item_type": "clinical_case",
        "question": draft["question"],
        "answer": draft["answer"],
        "options": None,
        "source_pdf": source["source_pdf"],
        "page_start": start,
        "page_end": end,
        "gold_evidence_units": [entry["quote"] for entry in evidence],
        "evidence_note": (
            f"Source-adapted vignette: {draft['case']}; physical PDF pages {start}-{end}. "
            "Supporting excerpts and rationale are in metadata. Physician review pending."
        ),
        "topic": draft["topic"],
        "section_label": draft["case"],
        "difficulty": "unrated",
        "acceptable_answers": [draft["answer"]]
        + config["answer_aliases"].get(draft["answer"], []),
        "hallucination_red_flags": flags,
        "metadata": metadata,
    }
