"""Exercise real inference and result state through Streamlit's test runner."""

from pathlib import Path
import unittest
from unittest.mock import patch

import joblib
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["cycle", "voltage", "temperature", "capacity"]


def html_text(app):
    return "\n".join(node.proto.body for node in app.get("html"))


class PredictionAppTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
        self.assertEqual(len(self.app.exception), 0)

    def predict(self):
        self.app.button[0].click().run()
        self.assertEqual(len(self.app.exception), 0)
        self.assertEqual(len(self.app.error), 0)
        return self.app.session_state["prediction"]

    def test_prediction_matches_saved_pipeline_and_percent_display(self):
        self.assertIn("No prediction yet", html_text(self.app))
        result = self.predict()
        model = joblib.load(ROOT / "models/random_forest_pipeline.joblib")
        frame = pd.DataFrame([result["inputs"]], columns=FEATURES)
        expected = float(model.predict(frame)[0])
        self.assertAlmostEqual(result["soh"], expected, places=12)
        self.assertIn(f"{expected * 100:.1f} percent", html_text(self.app))
        self.assertEqual(len(self.app.get("download_button")), 1)

    def test_edit_hides_stale_result_and_export_until_recalculated(self):
        previous = self.predict()
        self.app.number_input(key="capacity").set_value(1.2).run()
        self.assertIn("Measurements updated", html_text(self.app))
        self.assertIn("No prediction yet", html_text(self.app))
        self.assertEqual(len(self.app.get("download_button")), 0)
        current = self.predict()
        self.assertEqual(current["inputs"]["capacity"], 1.2)
        self.assertNotEqual(current["soh"], previous["soh"])

    def test_sample_is_a_real_record_and_reset_clears_prediction(self):
        self.app.button[1].click().run()
        values = {feature: self.app.number_input(key=feature).value for feature in FEATURES}
        dataset = pd.read_csv(ROOT / "data/dataset.csv")
        self.assertTrue(dataset[FEATURES].eq(pd.Series(values)).all(axis=1).any())
        self.assertEqual(len(self.app.warning), 0)
        self.predict()
        self.app.button[2].click().run()
        self.assertIn("No prediction yet", html_text(self.app))
        self.assertEqual(len(self.app.get("download_button")), 0)
        self.assertEqual(self.app.number_input(key="capacity").value, 1.75)

    def test_outside_range_warns_but_still_returns_real_prediction(self):
        self.app.number_input(key="cycle").set_value(500).run()
        self.assertIn("Outside the training range", self.app.warning[0].value)
        result = self.predict()
        self.assertEqual(result["inputs"]["cycle"], 500)

    def test_model_loading_failure_is_a_readable_error(self):
        st.cache_resource.clear()
        with patch("joblib.load", side_effect=FileNotFoundError("Missing model")):
            with self.assertLogs(level="ERROR"):
                failed = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
        self.assertEqual(len(failed.exception), 0)
        self.assertIn("prediction resources could not be loaded", failed.error[0].value)
        self.assertEqual(len(failed.number_input), 0)
        st.cache_resource.clear()


if __name__ == "__main__":
    unittest.main()
