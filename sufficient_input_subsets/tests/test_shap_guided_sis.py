# coding=utf-8
"""Tests for SHAP-guided SIS acceleration."""

from __future__ import annotations

import unittest

import numpy as np

from sufficient_input_subsets import sis
from sufficient_input_subsets.shap_guided_sis import (
    collection_mask,
    shap_guided_sis_collection,
)


def f_sum(batch):
  batch = np.asarray(batch)
  return np.array([np.sum(example) for example in batch], dtype=float)


class ShapGuidedSISTest(unittest.TestCase):

  def test_returns_sufficient_subset_when_one_exists(self):
    x = np.array([0.2, 0.7, 0.6])
    masked = np.zeros_like(x)
    threshold = 1.0

    collection, diagnostics = shap_guided_sis_collection(
        f_sum, threshold, x, masked, batch_size=8)
    mask = collection_mask(collection)
    score = float(f_sum(sis.produce_masked_inputs(x, masked, np.asarray([mask])))[0])

    self.assertTrue(collection)
    self.assertGreaterEqual(score, threshold)
    self.assertTrue(diagnostics["threshold_met"])

  def test_uses_no_more_calls_than_naive_feature_scoring(self):
    x = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1])
    masked = np.zeros_like(x)

    _, diagnostics = shap_guided_sis_collection(
        f_sum, 1.4, x, masked, batch_size=64)

    naive_unbatched_calls = 1 + x.size + x.size + x.size
    self.assertLessEqual(
        diagnostics["model_call_count"], naive_unbatched_calls)

  def test_handles_2d_inputs(self):
    x = np.array([[0.4, 0.8], [0.1, 0.3]])
    masked = np.zeros_like(x)
    collection, _ = shap_guided_sis_collection(f_sum, 1.0, x, masked)

    mask = collection_mask(collection)
    self.assertEqual(mask.shape, x.shape)
    score = float(f_sum(sis.produce_masked_inputs(x, masked, np.asarray([mask])))[0])
    self.assertGreaterEqual(score, 1.0)

  def test_return_diagnostics_false_matches_original_shape(self):
    x = np.array([1.0, 0.2, 0.2])
    masked = np.zeros_like(x)
    collection = shap_guided_sis_collection(
        f_sum, 0.9, x, masked, return_diagnostics=False)

    self.assertIsInstance(collection, list)
    self.assertTrue(all(hasattr(result, "mask") for result in collection))


if __name__ == "__main__":
  unittest.main()
