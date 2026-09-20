"""Transparent score mappings and conservative MCQ parsing."""
import re
import numpy as np


def parse_mcq(text, options):
    """Require an anchored option marker; an initial article 'A' is not an answer."""
    text = text.strip().replace("**", "").replace("__", "").replace("`", "")
    text = re.sub(r"^\s*#{1,6}\s*", "", text)
    prefix = r"(?:(?:the\s+)?(?:correct\s+|best\s+|selected\s+|final\s+)?(?:answer|option|choice)\s*(?:is\s*)?[:\-]?\s*)?"
    match = re.match(
        "^" + prefix + r"\(?([A-Z])\)?(?=[.\):\-]|[ \t]*(?:\n|$))", text, re.I
    )
    if not match or match.group(1).upper() not in options:
        return None
    return match.group(1).upper()


def strict_score(label):
    return {"correct": 1.0, "partial": 0.0, "incorrect": 0.0}.get(label, np.nan)


def partial_credit(label):
    return {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}.get(label, np.nan)


def criteria_coverage(statuses):
    values = {"covered": 1.0, "partial": 0.5, "missing": 0.0, "contradicted": 0.0}
    if not statuses or any(status not in values for status in statuses):
        return np.nan
    return sum(values[status] for status in statuses) / len(statuses)
