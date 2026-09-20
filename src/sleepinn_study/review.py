"""Human question review with a separate, append-only decision history."""
from copy import deepcopy
from datetime import datetime, timezone
from html import escape
from pathlib import Path
import json
from uuid import uuid4

from .databank import validate_items
from .io import digest, read_json, write_json


class ReviewSession:
    def __init__(self, items, destination):
        validate_items(items)
        if not items:
            raise ValueError("Choose a nonempty question set")
        self.items = deepcopy(items)
        self.by_id = {item["item_id"]: item for item in self.items}
        self.destination = Path(destination)
        self.fingerprint = digest(json.dumps(self.items, sort_keys=True).encode())
        self.latest = {}
        manifest = self.destination / "session.json"
        if manifest.exists():
            if read_json(manifest)["input_sha256"] != self.fingerprint:
                raise ValueError("This review folder belongs to another input version")
            for path in sorted((self.destination / "decisions").glob("*.json")):
                record = read_json(path)
                if record["input_sha256"] != self.fingerprint:
                    raise ValueError("Review decision belongs to another input version")
                if record["item_id"] not in self.by_id or record["status"] not in {"accepted", "revised", "rejected"}:
                    raise ValueError("Invalid saved review decision")
                if record["original"] != self.by_id[record["item_id"]]:
                    raise ValueError("Saved decision does not match its original item")
                if record["status"] != "rejected":
                    validate_items([record["revised"]])
                self.latest[record["item_id"]] = record

    def current(self, item_id):
        record = self.latest.get(item_id)
        if record and record["status"] != "rejected":
            return deepcopy(record["revised"])
        return deepcopy(self.by_id[item_id])

    def _start(self):
        manifest = self.destination / "session.json"
        if not manifest.exists():
            write_json(
                manifest, {"input_sha256": self.fingerprint, "items": len(self.items)}
            )

    def save(self, item_id, edited, decision, reviewer, source_checked, notes=""):
        if decision not in {"accepted", "rejected", "revised"}:
            raise ValueError(
                "Choose Accept question, Reject question, or Save revision"
            )
        if not reviewer.strip():
            raise ValueError("Enter the human reviewer’s name or identifier")
        before = self.current(item_id)
        if edited["item_id"] != item_id or edited["item_type"] != before["item_type"]:
            raise ValueError("The question identity and type must stay the same")
        if decision == "accepted" and edited != before:
            raise ValueError(
                "There are edits: use Save revision, or Discard edits first"
            )
        if decision == "revised" and edited == before:
            raise ValueError("No fields changed; use Accept question instead")
        if decision != "rejected" and not source_checked:
            raise ValueError(
                "Check the cited source and tick the source checkbox first"
            )
        if decision in {"rejected", "revised"} and not notes.strip():
            raise ValueError("Add a short reason for the rejection or change")
        result = deepcopy(before if decision == "rejected" else edited)
        if decision != "rejected":
            validate_items([result])
            if result["item_type"] == "mcq":
                options = result["options"]
                if not isinstance(options, dict) or set(options) != set("ABCD"):
                    raise ValueError("An MCQ needs options A, B, C, and D")
                if any(not str(value).strip() for value in options.values()):
                    raise ValueError("Fill in all four MCQ options")
            if isinstance(result["answer"], list) and any(
                not isinstance(part, str) or not part.strip()
                for part in result["answer"]
            ):
                raise ValueError("Each answer component must contain text")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        record = {
            "item_id": item_id,
            "status": decision,
            "reviewer": reviewer.strip(),
            "source_checked": bool(source_checked),
            "notes": notes.strip(),
            "reviewed_utc": stamp,
            "input_sha256": self.fingerprint,
            "original": deepcopy(self.by_id[item_id]),
            "before": before,
            "revised": result,
            "review_type": "manual; reviewer qualifications not independently verified",
        }
        self._start()
        path = self.destination / "decisions" / f"{stamp}_{uuid4().hex}.json"
        write_json(path, record)
        self.latest[item_id] = record
        return path

    def export(self):
        """Export only explicitly accepted/changed questions; retain the full decision history."""
        pending = set(self.by_id) - set(self.latest)
        if pending:
            raise ValueError(f"Review all items before final export: {len(pending)} still pending")
        selected = []
        for item in self.items:
            record = self.latest.get(item["item_id"])
            if record and record["status"] in {"accepted", "revised"}:
                value = deepcopy(record["revised"])
                value.setdefault("metadata", {})["manual_review"] = {
                    key: record[key]
                    for key in ["reviewer", "status", "source_checked", "reviewed_utc"]
                }
                selected.append(value)
        if not selected:
            raise ValueError("Accept or save a changed question before exporting")
        self._start()
        path = self.destination / "exports" / f"reviewed_{uuid4().hex}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            for item in selected:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        write_json(path.with_suffix(".manifest.json"), {
            "input_sha256": self.fingerprint, "output_sha256": digest(path.read_bytes()),
            "reviewed_items": len(self.latest), "included_items": len(selected),
            "rejected_items": sum(r["status"] == "rejected" for r in self.latest.values()),
            "all_items_decided": True, "review_method": "human",
        })
        return path, len(selected)


class QuestionReviewer:
    """Jupyter controls for one question at a time. Opening the form writes nothing."""

    def __init__(self, items, destination):
        import ipywidgets as widgets

        self.session = ReviewSession(items, destination)
        self.position = 0
        self.loading = False
        wide = widgets.Layout(width="100%")
        self.reviewer = widgets.Text(
            description="Reviewer name",
            placeholder="Your name or identifier",
            layout=wide,
            style={"description_width": "initial"},
        )
        self.selector = widgets.Dropdown(
            options=[
                (f"{i+1}. {item['question'][:90]}", i) for i, item in enumerate(items)
            ],
            description="Question",
            layout=wide,
        )
        self.progress = widgets.HTML()
        self.source = widgets.HTML()
        self.original = widgets.HTML()
        self.question = widgets.Textarea(
            description="Question", layout=widgets.Layout(width="100%", height="100px")
        )
        self.answer_hint = widgets.HTML()
        self.answer = widgets.Textarea(
            description="Answer", layout=widgets.Layout(width="100%", height="120px")
        )
        self.alternatives = widgets.Textarea(description="Accepted variants", layout=wide, placeholder="One acceptable answer per line")
        self.rationale = widgets.Textarea(description="Rationale", layout=wide)
        self.options = {
            key: widgets.Text(description=f"Option {key}", layout=wide)
            for key in "ABCD"
        }
        self.option_box = widgets.VBox(list(self.options.values()))
        self.checked = widgets.Checkbox(
            description="I checked the cited source for this decision",
            indent=False,
            layout=wide,
        )
        self.notes = widgets.Textarea(
            description="Review note",
            placeholder="Why accept, reject, or change this question?",
            layout=wide,
        )
        self.message = widgets.HTML()
        self.accept = widgets.Button(
            description="Accept question", button_style="success"
        )
        self.reject = widgets.Button(
            description="Reject question", button_style="danger"
        )
        self.change = widgets.Button(
            description="Save revision", button_style="warning"
        )
        self.reset = widgets.Button(description="Discard edits")
        self.previous = widgets.Button(description="Previous question")
        self.next = widgets.Button(description="Next question")
        self.export_button = widgets.Button(
            description="Export final reviewed bank",
            layout=widgets.Layout(width="220px"),
        )
        original_box = widgets.Accordion(children=[self.original])
        original_box.set_title(0, "Before human review (read only)")
        original_box.selected_index = None
        self.widget = widgets.VBox(
            [
                widgets.HTML(
                    "<h3>Human review</h3><p>Read the question, check the cited source, and record your own decision. Accept the question, reject it, or edit it and save a revision. Your decisions are saved separately from the supplied databank.</p><p>The progress counter shows recorded human decisions. Opening this optional form does not call an AI model or save a review.</p>"
                ),
                self.reviewer,
                self.selector,
                self.progress,
                self.source,
                original_box,
                self.question,
                self.answer_hint,
                self.answer,
                self.option_box,
                self.alternatives,
                self.rationale,
                self.checked,
                self.notes,
                widgets.HBox(
                    [self.accept, self.reject, self.change, self.reset],
                    layout=widgets.Layout(flex_flow="row wrap"),
                ),
                widgets.HBox(
                    [self.previous, self.next, self.export_button],
                    layout=widgets.Layout(flex_flow="row wrap"),
                ),
                self.message,
            ],
            layout=wide,
        )
        self.accept.on_click(lambda _: self._save("accepted"))
        self.reject.on_click(lambda _: self._save("rejected"))
        self.change.on_click(lambda _: self._save("revised"))
        self.reset.on_click(lambda _: self._load())
        self.previous.on_click(lambda _: self._move(self.position - 1))
        self.next.on_click(lambda _: self._move(self.position + 1))
        self.export_button.on_click(lambda _: self._export())
        self.selector.observe(self._select, names="value")
        self._load()

    def _edited(self):
        item = deepcopy(self.base)
        item["question"] = self.question.value
        # Keep the original list if untouched, including its exact whitespace.
        if self.answer.value != self.answer_initial:
            item["answer"] = (
                self.answer.value.splitlines()
                if isinstance(self.base["answer"], list)
                else self.answer.value
            )
        if item["item_type"] == "mcq":
            item["options"] = {
                key: widget.value for key, widget in self.options.items()
            }
        if self.alternatives.value != self.alternatives_initial:
            item["acceptable_answers"] = [s.strip() for s in self.alternatives.value.splitlines() if s.strip()]
        if self.rationale.value != self.rationale_initial:
            item.setdefault("metadata", {})["answer_rationale"] = self.rationale.value.strip()
        return item

    def _dirty(self):
        return self._edited() != self.base or bool(self.notes.value.strip())

    def _status(self, text, error=False):
        self.message.value = (
            f'<p style="color:{"#9b1c1c" if error else "#175c36"}">{escape(text)}</p>'
        )

    def _load(self):
        item_id = self.session.items[self.position]["item_id"]
        self.base = self.session.current(item_id)
        item = self.base
        self.question.value = item["question"]
        self.answer_initial = (
            "\n".join(item["answer"])
            if isinstance(item["answer"], list)
            else item["answer"]
        )
        self.answer.value = self.answer_initial
        self.alternatives_initial = "\n".join(item.get("acceptable_answers") or [])
        self.alternatives.value = self.alternatives_initial
        self.rationale_initial = item.get("metadata", {}).get("answer_rationale", "")
        self.rationale.value = self.rationale_initial
        self.answer_hint.value = (
            "<p>Criteria: one reference component per line. Keep AND/OR qualifiers in the text.</p>"
            if isinstance(item["answer"], list)
            else "<p>MCQ: enter the correct option letter. Other questions: enter the complete reference answer. When changing an answer, also check accepted variants and the rationale below.</p>"
        )
        self.option_box.layout.display = "" if item["item_type"] == "mcq" else "none"
        for key, widget in self.options.items():
            widget.value = str((item.get("options") or {}).get(key, ""))
        self.checked.value = False
        self.notes.value = ""
        source = escape(str(item.get("source_pdf", "Not recorded")))
        evidence = "<br>".join(
            escape(str(q)) for q in item.get("gold_evidence_units", [])
        )
        numbering = escape(
            str(
                item.get("metadata", {}).get(
                    "page_numbering", "zero-based physical PDF indices"
                )
            )
        )
        self.source.value = f"<p><b>Source:</b> {source}<br><b>Stored pages:</b> {item.get('page_start')}–{item.get('page_end')} · {numbering}.<br><b>Evidence:</b><br>{evidence}</p><p>Check the actual PDF as well; the quotations alone are not a completed source review.</p>"
        original = self.session.by_id[item_id]
        self.original.value = (
            '<pre style="white-space:pre-wrap">'
            + escape(
                json.dumps(
                    {
                        key: original.get(key)
                        for key in [
                            "question",
                            "answer",
                            "options",
                            "acceptable_answers",
                            "hallucination_red_flags",
                            "evidence_note",
                        ]
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            + "</pre>"
        )
        saved = self.session.latest.get(item_id, {})
        self.progress.value = f"<p>Question {self.position+1} / {len(self.session.items)} · {len(self.session.latest)} human decisions recorded · Last decision: {escape(saved.get('status', 'not reviewed'))}</p>"
        self.previous.disabled = self.position == 0
        self.next.disabled = self.position == len(self.session.items) - 1
        self.message.value = ""

    def _select(self, change):
        if not self.loading:
            self._move(change["new"])

    def _move(self, position):
        if position == self.position:
            return
        if self._dirty():
            self.loading = True
            self.selector.value = self.position
            self.loading = False
            self._status(
                "Unsaved edits or notes. Save a decision, or press Discard edits before moving.",
                True,
            )
            return
        self.position = max(0, min(position, len(self.session.items) - 1))
        self.loading = True
        self.selector.value = self.position
        self.loading = False
        self._load()

    def _save(self, decision):
        try:
            path = self.session.save(
                self.base["item_id"],
                self._edited(),
                decision,
                self.reviewer.value,
                self.checked.value,
                self.notes.value,
            )
            self._load()
            self._status(
                f"Saved {decision} in {path.name}. Press Next question when ready."
            )
        except (ValueError, KeyError, TypeError, OSError) as error:
            self._status(str(error), True)

    def _export(self):
        if self._dirty():
            self._status("Save or reset the current edits before exporting.", True)
            return
        try:
            path, count = self.session.export()
            self._status(
                f"Exported {count} accepted/changed questions to {path}. All items have a decision; rejected questions are excluded."
            )
        except (ValueError, OSError) as error:
            self._status(str(error), True)
