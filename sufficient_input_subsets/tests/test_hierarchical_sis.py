# coding=utf-8
"""Tests for Hierarchical SIS."""

from __future__ import annotations

import unittest

import numpy as np

from sufficient_input_subsets.hierarchical_sis import (
    hierarchical_sis_collection,
    make_grid_feature_groups,
)


def f_sum(batch):
  return np.array([np.sum(example) for example in np.asarray(batch)], dtype=float)


class HierarchicalSISTest(unittest.TestCase):

  def test_returns_masks_at_multiple_levels(self):
    x = np.zeros((4, 4), dtype=float)
    x[0:2, 0:2] = 1.0
    result = hierarchical_sis_collection(
        f_sum, 2.0, x, np.zeros_like(x), levels=(2, 4), mode="pixel")

    self.assertGreaterEqual(len(result.level_masks), 2)
    self.assertEqual(result.level_masks[0].shape, x.shape)
    self.assertTrue(result.final_masks)
    self.assertIn("model_call_count", result.diagnostics)

  def test_grid_groups_handle_1d_inputs(self):
    groups = make_grid_feature_groups((6,), cells_per_axis=3)

    self.assertEqual(len(groups), 3)
    self.assertTrue(all(group.shape == (6,) for group in groups))

  def test_hierarchical_handles_1d_inputs(self):
    x = np.array([0.7, 0.6, 0.1, 0.1])
    result = hierarchical_sis_collection(
        f_sum, 1.0, x, np.zeros_like(x), levels=(2, 4), mode="grid")

    self.assertGreaterEqual(len(result.level_masks), 1)
    self.assertEqual(result.level_masks[-1].shape, x.shape)


if __name__ == "__main__":
  unittest.main()
