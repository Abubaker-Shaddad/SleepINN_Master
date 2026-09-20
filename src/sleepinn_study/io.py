"""Paths, atomic writes, and input fingerprints."""
from pathlib import Path
import hashlib
import json
import os


def project_root():
    return Path(os.environ.get("SLEEPINN_ROOT", Path(__file__).resolve().parents[2]))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    return digest(Path(path).read_bytes())


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def output_directory(name):
    """Keep new experiments away from the included study artifacts."""
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Use a single, descriptive experiment name")
    path = project_root() / "outputs" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def answer_frame(domain):
    import pandas as pd

    index = read_json(project_root() / "results/answers/INDEX.json")
    rows = []
    for entry in index:
        if entry["dataset"] == domain:
            rows.extend(read_jsonl(project_root() / entry["file"]))
    frame = pd.DataFrame(rows)
    keys = ["model_id", "precision", "item_id", "mode"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate answer identities")
    return frame
