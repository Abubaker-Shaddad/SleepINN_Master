import sys
from pathlib import Path
import tempfile
import unittest
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sleepinn_study.statistics import paired_effect, consensus_scores
from sleepinn_study.scoring import parse_mcq, criteria_coverage
from sleepinn_study.providers import request_json
from sleepinn_study.benchmark import prompt_jobs


class MethodsTests(unittest.TestCase):
    def test_missing_pair_is_not_zero(self):
        frame = pd.DataFrame(
            {
                "item_id": ["a", "a", "b", "b"],
                "mode": ["no_rag", "with_rag"] * 2,
                "score": [0, 1, 1, np.nan],
            }
        )
        result = paired_effect(frame, "score", planned_pairs=2, resamples=100)
        self.assertEqual(result["complete_pairs"], 1)
        self.assertEqual(result["delta_pp"], 100)
        self.assertEqual(result["full_case_lower_delta_pp"], 0)
        self.assertEqual(result["full_case_upper_delta_pp"], 100)

    def test_consensus_requires_same_valid_label(self):
        frame = pd.DataFrame(
            {
                "gemini_label": ["correct", "partial", "ungradable", None],
                "glm_label": ["correct", "correct", "ungradable", None],
            }
        )
        result = consensus_scores(frame)
        self.assertEqual(
            result.included_exact_consensus.tolist(), [True, False, False, False]
        )
        self.assertEqual(result.strict_score.notna().sum(), 1)

    def test_article_is_not_option(self):
        self.assertIsNone(
            parse_mcq("A patient usually experiences sleepiness", list("ABCD"))
        )
        self.assertEqual(
            parse_mcq("**Answer: B.** A sleep disorder.", list("ABCD")), "B"
        )

    def test_ungradable_component_is_missing(self):
        self.assertTrue(np.isnan(criteria_coverage(["covered", "ungradable"])))
        self.assertEqual(criteria_coverage(["covered", "partial", "missing"]), 0.5)

    def test_api_disabled_before_credentials_or_network(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                request_json("gemini", "unused", {}, Path(folder) / "request")
            self.assertFalse((Path(folder) / "request").exists())



if __name__ == "__main__":
    unittest.main()
