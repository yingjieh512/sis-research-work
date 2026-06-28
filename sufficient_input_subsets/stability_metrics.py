# coding=utf-8
"""Stability and robustness metrics for SIS explanations."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sufficient_input_subsets import sis


BatchFunction = Callable[[np.ndarray], np.ndarray]


def _as_bool_mask(mask: np.ndarray) -> np.ndarray:
  return np.asarray(mask, dtype=bool)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
  """Returns intersection-over-union between two boolean masks."""
  a = _as_bool_mask(mask_a).reshape(-1)
  b = _as_bool_mask(mask_b).reshape(-1)
  if a.shape != b.shape:
    raise ValueError("Masks must have the same number of elements.")
  intersection = np.logical_and(a, b).sum()
  union = np.logical_or(a, b).sum()
  if union == 0:
    return 1.0
  return float(intersection / union)


def mask_f1(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
  """Returns F1 overlap between two boolean masks."""
  a = _as_bool_mask(mask_a).reshape(-1)
  b = _as_bool_mask(mask_b).reshape(-1)
  if a.shape != b.shape:
    raise ValueError("Masks must have the same number of elements.")
  true_positive = np.logical_and(a, b).sum()
  false_positive = np.logical_and(~a, b).sum()
  false_negative = np.logical_and(a, ~b).sum()
  denominator = 2 * true_positive + false_positive + false_negative
  if denominator == 0:
    return 1.0
  return float((2 * true_positive) / denominator)


def explanation_stability(
    masks: Sequence[np.ndarray],
    metric: str = "iou",
) -> Dict[str, Any]:
  """Summarizes pairwise stability for a sequence of explanation masks."""
  if metric not in ("iou", "f1"):
    raise ValueError("metric must be 'iou' or 'f1'.")
  masks = [_as_bool_mask(mask) for mask in masks]
  n_masks = len(masks)
  pairwise = np.ones((n_masks, n_masks), dtype=float)
  scores: List[float] = []
  scorer = mask_iou if metric == "iou" else mask_f1

  for i in range(n_masks):
    for j in range(i + 1, n_masks):
      score = scorer(masks[i], masks[j])
      pairwise[i, j] = score
      pairwise[j, i] = score
      scores.append(score)

  subset_sizes = np.array([np.sum(mask) for mask in masks], dtype=float)
  return {
      "metric": metric,
      "mean_pairwise": float(np.mean(scores)) if scores else 1.0,
      "min_pairwise": float(np.min(scores)) if scores else 1.0,
      "max_pairwise": float(np.max(scores)) if scores else 1.0,
      "pairwise_matrix": pairwise,
      "subset_size_mean": float(np.mean(subset_sizes)) if n_masks else 0.0,
      "subset_size_variance": float(np.var(subset_sizes)) if n_masks else 0.0,
      "n_masks": n_masks,
  }


def _collection_union_mask(collection: Sequence[sis.SISResult]) -> np.ndarray:
  if not collection:
    return np.array([], dtype=bool)
  union = np.zeros_like(collection[0].mask, dtype=bool)
  for result in collection:
    union = np.logical_or(union, np.asarray(result.mask, dtype=bool))
  return union


def extract_explanation_mask(result: Any) -> np.ndarray:
  """Extracts one union mask from common SIS result formats."""
  if isinstance(result, tuple) and result:
    return extract_explanation_mask(result[0])
  if isinstance(result, list):
    return _collection_union_mask(result)
  if hasattr(result, "final_masks"):
    final_masks = getattr(result, "final_masks")
    if final_masks:
      union = np.zeros_like(final_masks[0], dtype=bool)
      for mask in final_masks:
        union = np.logical_or(union, np.asarray(mask, dtype=bool))
      return union
  if hasattr(result, "inclusion_probabilities"):
    return np.asarray(getattr(result, "inclusion_probabilities")) >= 0.5
  if hasattr(result, "mask"):
    return np.asarray(result.mask, dtype=bool)
  return np.asarray(result, dtype=bool)


def _call_sis_method(
    sis_method: Callable[..., Any],
    f_batch: BatchFunction,
    threshold: float,
    input_value: np.ndarray,
    fully_masked_input: np.ndarray,
) -> Any:
  return sis_method(f_batch, threshold, input_value, fully_masked_input)


def perturbation_stability(
    f_batch: BatchFunction,
    sis_method: Callable[..., Any],
    input_batch: np.ndarray,
    fully_masked_input: np.ndarray,
    perturb_fn: Callable[[np.ndarray], np.ndarray],
    threshold: float,
    n_perturbations: int = 10,
) -> Dict[str, Any]:
  """Measures explanation drift under repeated input perturbations.

  ``sis_method`` should accept ``(f_batch, threshold, input, fully_masked_input)``
  and may return a SIS collection, ``(collection, diagnostics)``, or one of the
  extension result objects.
  """
  inputs = np.asarray(input_batch)
  if inputs.ndim == np.asarray(fully_masked_input).ndim:
    inputs = np.expand_dims(inputs, axis=0)

  clean_masks: List[np.ndarray] = []
  perturbed_masks: List[np.ndarray] = []
  clean_scores: List[float] = []
  perturbed_scores: List[float] = []
  drift_scores: List[float] = []

  for input_value in inputs:
    masked_reference = np.asarray(fully_masked_input)
    if masked_reference.shape == inputs.shape:
      raise ValueError(
          "fully_masked_input should be one masked example, not a full batch.")

    clean_result = _call_sis_method(
        sis_method, f_batch, threshold, input_value, masked_reference)
    clean_mask = extract_explanation_mask(clean_result)
    clean_masks.append(clean_mask)
    clean_scores.append(float(np.asarray(f_batch(np.asarray([input_value]))).reshape(-1)[0]))

    for _ in range(n_perturbations):
      perturbed = np.asarray(perturb_fn(np.copy(input_value)))
      perturbed_result = _call_sis_method(
          sis_method, f_batch, threshold, perturbed, masked_reference)
      perturbed_mask = extract_explanation_mask(perturbed_result)
      perturbed_masks.append(perturbed_mask)
      perturbed_scores.append(
          float(np.asarray(f_batch(np.asarray([perturbed]))).reshape(-1)[0]))
      if clean_mask.size and perturbed_mask.size:
        drift_scores.append(1.0 - mask_iou(clean_mask, perturbed_mask))

  stability = explanation_stability(
      clean_masks + perturbed_masks, metric="iou") if (clean_masks or perturbed_masks) else {}
  return {
      "clean_masks": clean_masks,
      "perturbed_masks": perturbed_masks,
      "clean_scores": clean_scores,
      "perturbed_scores": perturbed_scores,
      "mean_explanation_drift": float(np.mean(drift_scores)) if drift_scores else 0.0,
      "max_explanation_drift": float(np.max(drift_scores)) if drift_scores else 0.0,
      "mean_confidence_retention": (
          float(np.mean(perturbed_scores) / np.mean(clean_scores))
          if clean_scores and np.mean(clean_scores) != 0 else float("nan")),
      "stability_summary": stability,
  }


def adversarial_sensitivity_score(
    clean_masks: Sequence[np.ndarray],
    perturbed_masks: Sequence[np.ndarray],
    clean_scores: Optional[Sequence[float]] = None,
    perturbed_scores: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
  """Scores explanation and confidence sensitivity under perturbation."""
  if len(clean_masks) == 0 or len(perturbed_masks) == 0:
    return {
        "mean_mask_iou": 1.0,
        "explanation_drift": 0.0,
        "confidence_drop": 0.0,
        "sensitivity_score": 0.0,
  }

  ious = []
  for clean_mask in clean_masks:
    for perturbed_mask in perturbed_masks:
      if np.asarray(clean_mask).size == np.asarray(perturbed_mask).size:
        ious.append(mask_iou(clean_mask, perturbed_mask))
  mean_iou = float(np.mean(ious)) if ious else 0.0
  explanation_drift = 1.0 - mean_iou

  confidence_drop = 0.0
  if clean_scores is not None and perturbed_scores is not None:
    clean_mean = float(np.mean(clean_scores))
    perturbed_mean = float(np.mean(perturbed_scores))
    confidence_drop = max(0.0, clean_mean - perturbed_mean)

  return {
      "mean_mask_iou": mean_iou,
      "explanation_drift": explanation_drift,
      "confidence_drop": confidence_drop,
      "sensitivity_score": float(explanation_drift + confidence_drop),
  }
