from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sleepinn_study.review import ReviewSession, QuestionReviewer


def questions():
    return [
        {
            "item_id": "one",
            "item_type": "free_text_qa",
            "question": "A question?",
            "answer": "A reference",
            "options": None,
        },
        {
            "item_id": "two",
            "item_type": "criteria_qa",
            "question": "Which components?",
            "answer": ["First", "Second"],
            "options": None,
        },
    ]


class ReviewTests(unittest.TestCase):
    def test_decisions_preserve_original_and_resume_export(self):
        with tempfile.TemporaryDirectory() as folder:
            items = questions()
            original = deepcopy(items)
            destination = Path(folder) / "review"
            session = ReviewSession(items, destination)
            self.assertFalse(destination.exists())
            with self.assertRaisesRegex(ValueError, "source"):
                session.save("one", items[0], "accepted", "Reviewer", False)
            session.save("one", items[0], "accepted", "Reviewer", True)
            edit = deepcopy(items[1])
            edit["answer"][1] = "A corrected component"
            with self.assertRaisesRegex(ValueError, "There are edits"):
                session.save("two", edit, "accepted", "Reviewer", True)
            session.save(
                "two", edit, "revised", "Reviewer", True, "Checked the wording"
            )
            resumed = ReviewSession(items, destination)
            self.assertEqual(resumed.current("two")["answer"], edit["answer"])
            self.assertEqual(resumed.export()[1], 2)
            resumed.save(
                "one", items[0], "rejected", "Reviewer", False, "Reference needs review"
            )
            self.assertEqual(resumed.export()[1], 1)
            self.assertEqual(len(list((destination / "decisions").glob("*.json"))), 3)
            self.assertEqual(items, original)
            with self.assertRaisesRegex(ValueError, "another input"):
                ReviewSession([edit, items[0]], destination)

    def test_widget_buttons_and_unsaved_navigation(self):
        with tempfile.TemporaryDirectory() as folder:
            widget = QuestionReviewer(questions(), Path(folder) / "review")
            widget.question.value = "Edited question?"
            widget.next.click()
            self.assertEqual(widget.position, 0)
            self.assertIn("Unsaved", widget.message.value)
            widget.reviewer.value = "Reviewer"
            widget.checked.value = True
            widget.notes.value = "Clearer wording"
            widget.change.click()
            self.assertIn("Saved revised", widget.message.value)
            widget.next.click()
            self.assertEqual(widget.position, 1)
            widget.notes.value = "Reference missing a component"
            widget.reject.click()
            self.assertIn("Saved rejected", widget.message.value)
            widget.previous.click()
            self.assertEqual(widget.question.value, "Edited question?")
            widget.export_button.click()
            self.assertIn("Exported 1", widget.message.value)
            widget.widget.close()


if __name__ == "__main__":
    unittest.main()
