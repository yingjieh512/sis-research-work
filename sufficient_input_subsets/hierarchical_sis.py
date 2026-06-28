# coding=utf-8
"""Hierarchical, multi-scale Sufficient Input Subsets."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sufficient_input_subsets import sis
from sufficient_input_subsets.shap_guided_sis import (
    collection_mask,
    make_singleton_feature_groups,
    shap_guided_sis_collection,
)


@dataclasses.dataclass
class HierarchicalSISResult:
  """Container returned by ``hierarchical_sis_collection``."""

  tree: List[Dict[str, Any]]
  level_masks: List[np.ndarray]
  final_masks: List[np.ndarray]
  diagnostics: Dict[str, Any]


def _axis_edges(length: int, cells: int) -> np.ndarray:
  cells = max(1, min(int(cells), int(length)))
  edges = np.linspace(0, length, cells + 1, dtype=int)
  return np.unique(edges)


def make_grid_feature_groups(
    mask_shape: Sequence[int],
    cells_per_axis: int,
    active_mask: Optional[np.ndarray] = None,
    min_region_size: int = 1,
) -> List[np.ndarray]:
  """Creates grid-cell feature groups over 1-D or image-like masks."""
  mask_shape = tuple(int(dim) for dim in mask_shape)
  if active_mask is None:
    active_mask = np.ones(mask_shape, dtype=bool)
  else:
    active_mask = np.asarray(active_mask, dtype=bool)

  groups: List[np.ndarray] = []
  if len(mask_shape) == 1:
    edges = _axis_edges(mask_shape[0], cells_per_axis)
    for start, end in zip(edges[:-1], edges[1:]):
      group = np.zeros(mask_shape, dtype=bool)
      group[start:end] = True
      group = np.logical_and(group, active_mask)
      if np.sum(group) >= min_region_size:
        groups.append(group)
    return groups

  height, width = mask_shape[0], mask_shape[1]
  row_edges = _axis_edges(height, cells_per_axis)
  col_edges = _axis_edges(width, cells_per_axis)
  for row_start, row_end in zip(row_edges[:-1], row_edges[1:]):
    for col_start, col_end in zip(col_edges[:-1], col_edges[1:]):
      group = np.zeros(mask_shape, dtype=bool)
      if len(mask_shape) == 2:
        group[row_start:row_end, col_start:col_end] = True
      else:
        group[row_start:row_end, col_start:col_end, ...] = True
      group = np.logical_and(group, active_mask)
      if np.sum(group) >= min_region_size:
        groups.append(group)
  return groups


def _make_superpixel_groups_or_none(
    input_value: np.ndarray,
    active_mask: np.ndarray,
    n_segments: int,
    min_region_size: int,
) -> Optional[List[np.ndarray]]:
  """Optionally creates superpixel groups if scikit-image is installed."""
  try:
    from skimage.segmentation import slic  # type: ignore
  except Exception:
    return None

  image = np.asarray(input_value)
  if image.ndim < 2:
    return None
  try:
    labels = slic(
        image,
        n_segments=max(2, int(n_segments)),
        compactness=10.0,
        start_label=0,
        channel_axis=-1 if image.ndim == 3 else None)
  except Exception:
    return None

  groups = []
  for label in np.unique(labels):
    group = labels == label
    if active_mask.ndim > group.ndim:
      group = np.expand_dims(group, axis=-1)
      group = np.broadcast_to(group, active_mask.shape)
    group = np.logical_and(group, active_mask)
    if np.sum(group) >= min_region_size:
      groups.append(group)
  return groups


def _union_collection_or_previous(
    collection: Sequence[sis.SISResult],
    previous_mask: np.ndarray,
) -> np.ndarray:
  if collection:
    return collection_mask(collection)
  return np.asarray(previous_mask, dtype=bool)


def hierarchical_sis_collection(
    f_batch,
    threshold,
    initial_input,
    fully_masked_input,
    levels=(4, 8, 16),
    mode="grid",
    min_region_size=1,
    use_shap_guidance=True,
    return_tree=True,
):
  """Runs SIS over a coarse-to-fine feature hierarchy.

  Args:
    f_batch: Batched black-box function returning one scalar per input.
    threshold: Sufficiency threshold.
    initial_input: Input to explain.
    fully_masked_input: Baseline/masked input.
    levels: Grid resolutions. For 2-D inputs, each level is cells per axis.
    mode: ``"grid"``, ``"pixel"``, or optional ``"superpixel"``.
    min_region_size: Minimum active mask positions for a region.
    use_shap_guidance: Use the SHAP-guided wrapper at each level.
    return_tree: Included for API clarity; the returned object always stores
      the tree, but diagnostics records this flag.

  Returns:
    ``HierarchicalSISResult`` containing level masks, final masks, and tree.
  """
  initial_input = np.asarray(initial_input)
  fully_masked_input = np.asarray(fully_masked_input)
  if mode not in ("grid", "pixel", "superpixel"):
    raise ValueError("mode must be 'grid', 'pixel', or 'superpixel'.")
  if not levels:
    raise ValueError("levels must contain at least one resolution.")

  previous_mask = sis.make_empty_boolean_mask(initial_input.shape)
  tree: List[Dict[str, Any]] = []
  level_masks: List[np.ndarray] = []
  total_calls = 0
  total_evaluations = 0
  fallback_notes: List[str] = []

  for level_index, level in enumerate(levels):
    last_level = level_index == len(levels) - 1
    if mode == "pixel" and last_level:
      feature_groups = make_singleton_feature_groups(previous_mask)
      level_name = "pixel"
    elif mode == "superpixel":
      feature_groups = _make_superpixel_groups_or_none(
          initial_input, previous_mask, int(level), int(min_region_size))
      if feature_groups is None:
        fallback_notes.append(
            "superpixel mode unavailable; fell back to grid at level %s" % level)
        feature_groups = make_grid_feature_groups(
            previous_mask.shape, int(level), previous_mask, int(min_region_size))
      level_name = "superpixel"
    else:
      feature_groups = make_grid_feature_groups(
          previous_mask.shape, int(level), previous_mask, int(min_region_size))
      level_name = "grid"

    if not feature_groups:
      feature_groups = make_singleton_feature_groups(previous_mask)

    if use_shap_guidance:
      collection, diagnostics = shap_guided_sis_collection(
          f_batch,
          threshold,
          initial_input,
          fully_masked_input,
          initial_mask=previous_mask,
          feature_groups=feature_groups,
          importance_method="perturbation",
          return_diagnostics=True)
    else:
      collection = sis.sis_collection(
          f_batch,
          threshold,
          initial_input,
          fully_masked_input,
          initial_mask=previous_mask)
      diagnostics = {
          "model_call_count": None,
          "individual_model_evaluations": None,
          "threshold_met": bool(collection),
      }

    selected_mask = _union_collection_or_previous(collection, previous_mask)
    level_masks.append(selected_mask)
    total_calls += int(diagnostics.get("model_call_count", 0) or 0)
    total_evaluations += int(
        diagnostics.get("individual_model_evaluations", 0) or 0)
    tree.append({
        "level_index": level_index,
        "level": int(level),
        "mode": level_name,
        "n_groups": len(feature_groups),
        "selected_feature_count": int(np.sum(selected_mask)),
        "collection": collection,
        "diagnostics": diagnostics,
    })

    previous_mask = selected_mask
    if not np.any(previous_mask):
      break

  final_masks = [result.mask for result in tree[-1]["collection"]] if tree else []
  if not final_masks and level_masks:
    final_masks = [level_masks[-1]]

  diagnostics = {
      "levels": tuple(int(level) for level in levels),
      "mode": mode,
      "min_region_size": int(min_region_size),
      "use_shap_guidance": bool(use_shap_guidance),
      "return_tree": bool(return_tree),
      "model_call_count": int(total_calls),
      "individual_model_evaluations": int(total_evaluations),
      "n_levels_returned": len(level_masks),
      "fallback_notes": fallback_notes,
  }
  return HierarchicalSISResult(
      tree=tree,
      level_masks=level_masks,
      final_masks=final_masks,
      diagnostics=diagnostics,
  )


def plot_hierarchical_masks(level_masks: Sequence[np.ndarray], axes=None):
  """Plots one mask per hierarchy level."""
  try:
    import matplotlib.pyplot as plt
  except Exception as exc:
    raise ImportError("matplotlib is required for plotting.") from exc

  n_levels = len(level_masks)
  if axes is None:
    _, axes = plt.subplots(1, n_levels, figsize=(3 * n_levels, 3))
  if n_levels == 1:
    axes = [axes]
  for idx, (mask, ax) in enumerate(zip(level_masks, axes)):
    image = np.asarray(mask, dtype=float)
    if image.ndim == 1:
      image = np.expand_dims(image, axis=0)
    elif image.ndim > 2:
      image = np.mean(image, axis=-1)
    ax.imshow(image, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title("Level %d" % idx)
    ax.set_xticks([])
    ax.set_yticks([])
  return axes
