# coding=utf-8
"""Tests for Probabilistic SIS."""

from __future__ import annotations

import unittest

import numpy as np

from sufficient_input_subsets.probabilistic_sis import (
    probabilistic_sis_collection,
    top_k_stable_features,
)


def f_sum(batch):
  return np.array([np.sum(example) for example in np.asarray(batch)], dtype=float)


class ProbabilisticSISTest(unittest.TestCase):

  def test_inclusion_probabilities_are_valid(self):
    x = np.array([0.8, 0.7, 0.1])
    result = probabilistic_sis_collection(
        f_sum, 1.0, x, np.zeros_like(x), n_samples=5, random_state=1)

    self.assertEqual(result.inclusion_probabilities.shape, x.shape)
    self.assertTrue(np.all(result.inclusion_probabilities >= 0.0))
    self.assertTrue(np.all(result.inclusion_probabilities <= 1.0))
    self.assertGreaterEqual(result.mean_subset_size, 0.0)

  def test_handles_2d_inputs(self):
    x = np.array([[0.8, 0.2], [0.7, 0.1]])
    result = probabilistic_sis_collection(
        f_sum, 1.0, x, np.zeros_like(x), n_samples=4, random_state=2)

    self.assertEqual(result.inclusion_probabilities.shape, x.shape)
    self.assertIn("mean_pairwise", result.stability_summary)

  def test_top_k_stable_features(self):
    probabilities = np.array([0.1, 0.9, 0.3])
    table = top_k_stable_features(probabilities, k=2)

    if hasattr(table, "iloc"):
      self.assertEqual(tuple(table.iloc[0]["index"]), (1,))
    else:
      self.assertEqual(table[0]["index"], (1,))


if __name__ == "__main__":
  unittest.main()
