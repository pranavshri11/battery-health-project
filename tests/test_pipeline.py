"""Regression checks for invalid data, leakage boundaries, and saved inference."""

import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

from src.data_preprocessing import FEATURES, ROOT, dataset_sha256, load_dataset, validate_dataset
from src.feature_engineering import make_splits, split_manifest
from src.model_training import make_pipeline, metrics
from src.prediction import load_predictor, predict_soh, validate_inputs


class DataAndSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_dataset()

    def test_real_dataset_contract_and_derived_labels(self):
        self.assertEqual(self.data.shape, (1415, 7))
        self.assertEqual(self.data.battery_id.nunique(), 34)
        initial = self.data.groupby("battery_id").capacity.first()
        np.testing.assert_allclose(self.data.soh, self.data.capacity / self.data.battery_id.map(initial))
        self.assertGreater(int((self.data.soh > 1).sum()), 0)

    def test_dataset_checksum_survives_windows_line_endings(self):
        source = ROOT / "data/dataset.csv"
        with tempfile.TemporaryDirectory() as folder:
            crlf = Path(folder) / "dataset.csv"
            crlf.write_bytes(source.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
            self.assertEqual(dataset_sha256(source), dataset_sha256(crlf))

    def test_invalid_data_fails_instead_of_silent_cleaning(self):
        invalid = [self.data.drop(columns="soh"), pd.concat([self.data, self.data.iloc[:1]])]
        for key, value in [("capacity", -1), ("temperature", -280), ("voltage", np.inf),
                           ("soh", np.nan), ("cycle", 1.5), ("battery_id", " ")]:
            frame = self.data.copy()
            if key == "cycle":
                frame[key] = frame[key].astype(float)
            frame.loc[0, key] = value
            invalid.append(frame)
        for frame in invalid:
            with self.subTest(columns=list(frame.columns)):
                with self.assertRaises(ValueError):
                    validate_dataset(frame)

    def test_holdout_groups_and_every_cv_fold_are_disjoint(self):
        splits = make_splits(self.data)
        train = self.data.iloc[splits["development"]]
        test = self.data.iloc[splits["unseen_batteries"]]
        self.assertFalse(set(train.battery_id) & set(test.battery_id))
        self.assertEqual(len(train) + len(test), len(self.data))
        early = self.data.iloc[splits["development_early"]]
        for tr, va in GroupKFold(5).split(early[FEATURES], early.soh, early.battery_id):
            self.assertFalse(set(early.iloc[tr].battery_id) & set(early.iloc[va].battery_id))
        self.assertFalse(split_manifest(self.data, splits).partition.eq("").any())

    def test_later_cycles_never_enter_temporal_training(self):
        splits = make_splits(self.data)
        early = self.data.iloc[splits["development_early"]]
        later = self.data.iloc[splits["development_later"]]
        self.assertFalse(set(early.index) & set(later.index))
        for battery, part in later.groupby("battery_id"):
            self.assertLess(early.loc[early.battery_id == battery, "cycle"].max(), part.cycle.min())

    def test_scaler_is_fitted_only_on_the_passed_training_subset(self):
        splits = make_splits(self.data)
        early = self.data.iloc[splits["development_early"]]
        tr, va = next(GroupKFold(5).split(early[FEATURES], early.soh, early.battery_id))
        model = make_pipeline(RandomForestRegressor(n_estimators=5, random_state=42))
        model.fit(early.iloc[tr][FEATURES], early.iloc[tr].soh)
        np.testing.assert_allclose(model.named_steps["scaler"].mean_, early.iloc[tr][FEATURES].mean())
        self.assertFalse(np.allclose(model.named_steps["scaler"].mean_, early[FEATURES].mean()))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.joblib"
            joblib.dump(model, path)
            np.testing.assert_array_equal(model.predict(early.iloc[va][FEATURES]), joblib.load(path).predict(early.iloc[va][FEATURES]))


class InferenceTests(unittest.TestCase):
    def test_reorders_features_and_rejects_target_columns(self):
        values = {"capacity": 1.5, "temperature": 25, "voltage": 3.5, "cycle": 42}
        self.assertEqual(list(validate_inputs(values).columns), FEATURES)
        for key, value in [("capacity", 0), ("cycle", 2.2), ("voltage", np.nan), ("temperature", -300)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_inputs({**values, key: value})
        with self.assertRaises(ValueError):
            validate_inputs({**values, "soh": 0.9})

    def test_saved_model_metrics_and_metadata_are_reproducible(self):
        model, metadata = load_predictor()
        data = load_dataset()
        subset = data[data.battery_id.isin(metadata["holdout_batteries"])]
        self.assertFalse(set(metadata["development_batteries"]) & set(metadata["holdout_batteries"]))
        measured = metrics(subset.soh, predict_soh(model, subset[FEATURES]))
        for key in ["MAE", "MSE", "RMSE", "R2"]:
            self.assertAlmostEqual(measured[key], metadata["metrics"]["rf_unseen_batteries"][key], places=12)

    def test_mismatched_model_metadata_is_rejected(self):
        metadata = json.loads((ROOT / "models/model_metadata.json").read_text())
        metadata["artifact_sha256"] = "incorrect"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "metadata.json"
            path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_predictor(metadata_path=path)


if __name__ == "__main__":
    unittest.main()
