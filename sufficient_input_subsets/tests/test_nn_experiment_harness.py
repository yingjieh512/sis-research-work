# coding=utf-8
"""Tests for the neural-network SIS experiment harness."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments import run_nn_sis_experiments as harness
from sufficient_input_subsets import sis


class FakeProbabilisticModel:
  """Small predict_proba model for shape and scoring tests."""

  def predict_proba(self, flat_batch):
    flat_batch = np.asarray(flat_batch, dtype=float)
    score = np.clip(np.sum(flat_batch, axis=1) / max(flat_batch.shape[1], 1), 0.0, 1.0)
    return np.column_stack([1.0 - score, score])


class NNExperimentHarnessTest(unittest.TestCase):

  def test_digits_dataset_loading(self):
    dataset = harness.load_digits_dataset(seed=0)

    self.assertEqual(dataset.name, "digits")
    self.assertEqual(dataset.input_shape, (8, 8))
    self.assertEqual(dataset.n_classes, 10)
    self.assertEqual(dataset.x_train.ndim, 3)
    self.assertGreaterEqual(float(dataset.x_train.min()), 0.0)
    self.assertLessEqual(float(dataset.x_train.max()), 1.0)

  def test_counting_scoring_function_shapes(self):
    model = FakeProbabilisticModel()
    scorer = harness.CountingScoringFunction(model, target_class=1, input_shape=(2, 2))
    single = np.ones((2, 2), dtype=float)
    batch = np.stack([single, np.zeros((2, 2), dtype=float)])

    single_score = scorer(single)
    batch_scores = scorer(batch)

    self.assertEqual(single_score.shape, (1,))
    self.assertEqual(batch_scores.shape, (2,))
    self.assertEqual(scorer.batch_calls, 2)
    self.assertEqual(scorer.individual_model_evaluations, 3)

  def test_sis_method_threshold_satisfying_tiny_example(self):
    model = FakeProbabilisticModel()
    scorer = harness.CountingScoringFunction(model, target_class=1, input_shape=(2, 2))
    x = np.ones((2, 2), dtype=float)
    baseline = np.zeros_like(x)
    collection = sis.sis_collection(scorer, 0.5, x, baseline)
    mask = harness._collection_union_mask(collection, x.shape)
    final_score = harness._score_mask(scorer, x, baseline, mask)

    self.assertTrue(collection)
    self.assertGreaterEqual(final_score, 0.5)

  def test_results_writing(self):
    rows = [{
        "dataset": "toy",
        "method": "original_sis",
        "threshold_satisfied": True,
        "final_confidence": 0.9,
    }]
    metadata = {"dataset": {"name": "toy"}, "honesty_note": "measured"}
    with tempfile.TemporaryDirectory() as tmpdir:
      csv_path, json_path = harness.write_outputs(rows, metadata, Path(tmpdir))
      payload = json.loads(json_path.read_text(encoding="utf-8"))

    self.assertTrue(csv_path.name.endswith(".csv"))
    self.assertEqual(payload["metadata"]["dataset"]["name"], "toy")
    self.assertEqual(payload["results"][0]["method"], "original_sis")


if __name__ == "__main__":
  unittest.main()
