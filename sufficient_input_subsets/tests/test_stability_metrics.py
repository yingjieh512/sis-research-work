# coding=utf-8
"""Tests for SIS stability metrics."""

from __future__ import annotations

import unittest

import numpy as np

from sufficient_input_subsets.stability_metrics import (
    adversarial_sensitivity_score,
    explanation_stability,
    mask_f1,
    mask_iou,
    perturbation_stability,
)


def f_sum(batch):
  return np.array([np.sum(example) for example in np.asarray(batch)], dtype=float)


def simple_sis_method(f_batch, threshold, initial_input, fully_masked_input):
  del f_batch, threshold, fully_masked_input
  mask = np.zeros_like(initial_input, dtype=bool)
  mask.flat[int(np.argmax(initial_input))] = True
  return mask


class StabilityMetricsTest(unittest.TestCase):

  def test_mask_iou_and_f1_known_values(self):
    mask_a = np.array([True, True, False])
    mask_b = np.array([True, False, True])

    self.assertAlmostEqual(mask_iou(mask_a, mask_b), 1.0 / 3.0)
    self.assertAlmostEqual(mask_f1(mask_a, mask_b), 0.5)

  def test_explanation_stability_summary(self):
    masks = [
        np.array([True, False, False]),
        np.array([True, False, False]),
        np.array([False, True, False]),
    ]
    summary = explanation_stability(masks, metric="iou")

    self.assertEqual(summary["n_masks"], 3)
    self.assertGreaterEqual(summary["mean_pairwise"], 0.0)
    self.assertLessEqual(summary["mean_pairwise"], 1.0)
    self.assertAlmostEqual(summary["subset_size_variance"], 0.0)

  def test_perturbation_stability_runs(self):
    x = np.array([[0.9, 0.1], [0.2, 0.3]])

    def perturb_fn(value):
      return value

    summary = perturbation_stability(
        f_sum,
        simple_sis_method,
        np.asarray([x]),
        np.zeros_like(x),
        perturb_fn,
        threshold=0.5,
        n_perturbations=2)

    self.assertIn("mean_explanation_drift", summary)
    self.assertAlmostEqual(summary["mean_explanation_drift"], 0.0)

  def test_adversarial_sensitivity_score(self):
    clean = [np.array([True, False])]
    perturbed = [np.array([False, True])]
    score = adversarial_sensitivity_score(
        clean, perturbed, clean_scores=[0.9], perturbed_scores=[0.6])

    self.assertAlmostEqual(score["explanation_drift"], 1.0)
    self.assertAlmostEqual(score["confidence_drop"], 0.3)
    self.assertGreater(score["sensitivity_score"], 1.0)


if __name__ == "__main__":
  unittest.main()
