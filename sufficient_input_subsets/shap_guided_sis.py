# coding=utf-8
"""SHAP-guided acceleration wrappers for Sufficient Input Subsets.

This module builds on ``sufficient_input_subsets.sis`` instead of replacing it.
The public wrapper returns the same list-of-``SISResult`` collection by default,
with optional diagnostics describing runtime, model calls, and sufficiency.

The default importance estimator is SHAP-inspired but dependency-light: it
masks each feature or feature group, measures the drop in model confidence, and
uses that ranking to construct/prune a sufficient subset. If the optional
``shap`` package is installed, ``importance_method="shap"`` can use KernelSHAP
for singleton features on small inputs.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from sufficient_input_subsets import sis


ArrayLike = Union[np.ndarray, Sequence[float]]
BatchFunction = Callable[[np.ndarray], np.ndarray]


@dataclasses.dataclass
class FeatureImportanceResult:
  """Feature/group importance scores used to guide SIS search."""

  scores: np.ndarray
  values_when_masked: np.ndarray
  base_value: float
  ranking: np.ndarray
  feature_groups: List[np.ndarray]
  method: str
  batch_calls: int = 0
  individual_evaluations: int = 0


@dataclasses.dataclass
class MaskedModelEvaluator:
  """Batched, memoized evaluator for masked variants of one input."""

  f_batch: BatchFunction
  input_to_mask: np.ndarray
  fully_masked_input: np.ndarray
  batch_size: int = 64

  def __post_init__(self) -> None:
    self.cache: Dict[bytes, float] = {}
    self.batch_calls = 0
    self.individual_evaluations = 0

  @staticmethod
  def _cache_key(mask: np.ndarray) -> bytes:
    mask = np.ascontiguousarray(np.asarray(mask, dtype=bool))
    return mask.tobytes()

  def evaluate(self, mask: np.ndarray) -> float:
    """Evaluates one boolean mask."""
    return float(self.evaluate_many([mask])[0])

  def evaluate_many(self, masks: Sequence[np.ndarray]) -> np.ndarray:
    """Evaluates a sequence of boolean masks with memoization."""
    if not masks:
      return np.array([], dtype=float)

    normalized_masks = [np.asarray(mask, dtype=bool) for mask in masks]
    values = np.empty(len(normalized_masks), dtype=float)
    uncached_masks: List[np.ndarray] = []
    uncached_positions: List[int] = []

    for pos, mask in enumerate(normalized_masks):
      key = self._cache_key(mask)
      if key in self.cache:
        values[pos] = self.cache[key]
      else:
        uncached_masks.append(mask)
        uncached_positions.append(pos)

    for start in range(0, len(uncached_masks), max(1, self.batch_size)):
      chunk = uncached_masks[start:start + max(1, self.batch_size)]
      chunk_positions = uncached_positions[start:start + max(1, self.batch_size)]
      masked_inputs = sis.produce_masked_inputs(
          self.input_to_mask, self.fully_masked_input, np.asarray(chunk))
      chunk_values = np.asarray(self.f_batch(masked_inputs), dtype=float).reshape(-1)
      if chunk_values.shape[0] != len(chunk):
        raise ValueError(
            "f_batch must return one scalar value per masked input. "
            "Got %d values for %d inputs." % (chunk_values.shape[0], len(chunk)))

      self.batch_calls += 1
      self.individual_evaluations += len(chunk)
      for mask, pos, value in zip(chunk, chunk_positions, chunk_values):
        scalar = float(value)
        self.cache[self._cache_key(mask)] = scalar
        values[pos] = scalar

    return values


def _empty_mask_like(mask: np.ndarray) -> np.ndarray:
  return np.zeros(np.asarray(mask).shape, dtype=bool)


def _full_mask_like(mask: np.ndarray) -> np.ndarray:
  return np.ones(np.asarray(mask).shape, dtype=bool)


def _indices_to_mask(indices: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
  group = np.zeros(shape, dtype=bool)
  indices = np.asarray(indices, dtype=int)
  if indices.ndim == 1:
    indices = np.expand_dims(indices, axis=0)
  if indices.size:
    group[tuple(indices.T)] = True
  return group


def _normalize_feature_groups(
    feature_groups: Optional[Union[np.ndarray, Sequence[Any]]],
    mask_shape: Tuple[int, ...],
) -> Optional[List[np.ndarray]]:
  """Normalizes user-provided groups into boolean masks over ``mask_shape``."""
  if feature_groups is None:
    return None

  if isinstance(feature_groups, np.ndarray):
    arr = np.asarray(feature_groups)
    if arr.dtype == bool and arr.shape[1:] == mask_shape:
      return [np.asarray(group, dtype=bool) for group in arr]

  groups: List[np.ndarray] = []
  for raw_group in feature_groups:
    arr = np.asarray(raw_group)
    if arr.dtype == bool and arr.shape == mask_shape:
      groups.append(np.asarray(arr, dtype=bool))
    elif arr.ndim == 1 and arr.shape[0] == len(mask_shape):
      groups.append(_indices_to_mask(arr, mask_shape))
    elif arr.ndim == 2 and arr.shape[1] == len(mask_shape):
      groups.append(_indices_to_mask(arr, mask_shape))
    else:
      raise ValueError(
          "Each feature group must be a boolean mask with shape %s, a single "
          "coordinate, or an array of coordinates." % (mask_shape,))

  return groups


def make_singleton_feature_groups(active_mask: np.ndarray) -> List[np.ndarray]:
  """Creates one feature group per active position in a boolean mask."""
  active_mask = np.asarray(active_mask, dtype=bool)
  groups = []
  for coord in np.transpose(np.nonzero(active_mask)):
    group = np.zeros(active_mask.shape, dtype=bool)
    group[tuple(coord)] = True
    groups.append(group)
  return groups


def filter_feature_groups(
    feature_groups: Optional[Union[np.ndarray, Sequence[Any]]],
    active_mask: np.ndarray,
) -> List[np.ndarray]:
  """Returns non-empty feature groups restricted to currently active features."""
  active_mask = np.asarray(active_mask, dtype=bool)
  groups = _normalize_feature_groups(feature_groups, active_mask.shape)
  if groups is None:
    groups = make_singleton_feature_groups(active_mask)

  filtered = []
  for group in groups:
    restricted = np.logical_and(np.asarray(group, dtype=bool), active_mask)
    if np.any(restricted):
      filtered.append(restricted)
  return filtered


def _representative_index(group: np.ndarray) -> np.ndarray:
  coords = np.transpose(np.nonzero(group))
  if coords.size == 0:
    return np.zeros((group.ndim,), dtype=int)
  return coords[0].astype(int)


def _mask_to_sis_indices(mask: np.ndarray) -> np.ndarray:
  mask = np.asarray(mask, dtype=bool)
  coords = np.transpose(np.nonzero(mask)).astype(int)
  return coords.reshape((-1, mask.ndim))


def _as_ordering_array(indices: Sequence[np.ndarray], ndim: int) -> np.ndarray:
  if not indices:
    return np.empty((0, ndim), dtype=int)
  return np.asarray(indices, dtype=int).reshape((-1, ndim))


def _try_kernel_shap_scores(
    f_batch: BatchFunction,
    initial_input: np.ndarray,
    fully_masked_input: np.ndarray,
    current_mask: np.ndarray,
) -> Optional[FeatureImportanceResult]:
  """Computes optional KernelSHAP scores for small singleton-feature inputs."""
  if initial_input.size > 256 or current_mask.shape != initial_input.shape:
    return None

  try:
    import shap  # type: ignore
  except Exception:
    return None

  flat_input = np.asarray(initial_input, dtype=float).reshape(1, -1)
  flat_background = np.asarray(fully_masked_input, dtype=float).reshape(1, -1)
  input_shape = initial_input.shape
  counts = {"batch_calls": 0, "individual_evaluations": 0}

  def flat_model(flat_batch: np.ndarray) -> np.ndarray:
    batch = np.asarray(flat_batch, dtype=float).reshape((-1,) + input_shape)
    counts["batch_calls"] += 1
    counts["individual_evaluations"] += batch.shape[0]
    return np.asarray(f_batch(batch), dtype=float).reshape(-1)

  explainer = shap.KernelExplainer(flat_model, flat_background)
  shap_values = explainer.shap_values(
      flat_input, nsamples=min(2 * flat_input.shape[1] + 1, 256))
  if isinstance(shap_values, list):
    shap_values = shap_values[0]
  flat_scores = np.abs(np.asarray(shap_values, dtype=float).reshape(-1))

  groups = make_singleton_feature_groups(current_mask)
  active_coords = np.transpose(np.nonzero(current_mask))
  scores = np.array([flat_scores[np.ravel_multi_index(tuple(coord), input_shape)]
                     for coord in active_coords], dtype=float)
  ranking = np.argsort(-scores, kind="mergesort")
  base_value = float(flat_model(flat_input)[0])
  return FeatureImportanceResult(
      scores=scores,
      values_when_masked=np.full(scores.shape, np.nan, dtype=float),
      base_value=base_value,
      ranking=ranking,
      feature_groups=groups,
      method="shap",
      batch_calls=counts["batch_calls"],
      individual_evaluations=counts["individual_evaluations"],
  )


def estimate_feature_importance(
    f_batch: BatchFunction,
    initial_input: ArrayLike,
    fully_masked_input: ArrayLike,
    current_mask: Optional[np.ndarray] = None,
    feature_groups: Optional[Union[np.ndarray, Sequence[Any]]] = None,
    importance_method: str = "perturbation",
    batch_size: int = 64,
    evaluator: Optional[MaskedModelEvaluator] = None,
) -> FeatureImportanceResult:
  """Ranks features/groups by confidence drop when each group is masked.

  Args:
    f_batch: Batched black-box function returning one scalar per input.
    initial_input: Input being explained.
    fully_masked_input: Baseline/masked input.
    current_mask: Active positions that may still be used.
    feature_groups: Optional groups as boolean masks or coordinate arrays.
    importance_method: ``"perturbation"``, ``"shap"``, or ``"auto"``.
    batch_size: Batch size for perturbation scoring.
    evaluator: Optional memoized evaluator tied to this input.

  Returns:
    A ``FeatureImportanceResult`` with descending ``ranking`` indices.
  """
  initial_input = np.asarray(initial_input)
  fully_masked_input = np.asarray(fully_masked_input)
  if current_mask is None:
    current_mask = _full_mask_like(initial_input)
  else:
    current_mask = np.asarray(current_mask, dtype=bool)

  method = importance_method.lower()
  if method not in ("auto", "perturbation", "shap"):
    raise ValueError("importance_method must be 'auto', 'perturbation', or 'shap'.")

  normalized_groups = _normalize_feature_groups(feature_groups, current_mask.shape)
  if method in ("auto", "shap") and normalized_groups is None:
    shap_result = _try_kernel_shap_scores(
        f_batch, initial_input, fully_masked_input, current_mask)
    if shap_result is not None:
      return shap_result

  if evaluator is None:
    evaluator = MaskedModelEvaluator(
        f_batch, initial_input, fully_masked_input, batch_size=batch_size)

  groups = filter_feature_groups(normalized_groups, current_mask)
  base_value = evaluator.evaluate(current_mask)
  masked_variants = []
  for group in groups:
    masked_variants.append(np.logical_and(current_mask, ~group))

  values_when_masked = evaluator.evaluate_many(masked_variants)
  scores = base_value - values_when_masked
  ranking = np.argsort(-scores, kind="mergesort")
  return FeatureImportanceResult(
      scores=scores,
      values_when_masked=values_when_masked,
      base_value=base_value,
      ranking=ranking,
      feature_groups=groups,
      method="perturbation",
  )


def _ranked_find_sis(
    f_batch: BatchFunction,
    threshold: float,
    current_input: np.ndarray,
    current_mask: np.ndarray,
    fully_masked_input: np.ndarray,
    feature_groups: Optional[Union[np.ndarray, Sequence[Any]]] = None,
    importance_method: str = "perturbation",
    max_candidates: Optional[int] = None,
    batch_size: int = 64,
    ranking_noise_scale: float = 0.0,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> Tuple[Optional[sis.SISResult], Dict[str, Any]]:
  """Finds one sufficient subset using ranked construction and pruning."""
  evaluator = MaskedModelEvaluator(
      f_batch, current_input, fully_masked_input, batch_size=batch_size)
  start_value = evaluator.evaluate(current_mask)
  diagnostics: Dict[str, Any] = {
      "starting_score": start_value,
      "threshold": float(threshold),
      "threshold_met": bool(start_value >= threshold),
      "importance_method": importance_method,
  }
  if start_value < threshold or not np.any(current_mask):
    diagnostics.update({
        "sufficiency_score": start_value,
        "selected_feature_count": 0,
        "model_call_count": evaluator.batch_calls,
        "individual_model_evaluations": evaluator.individual_evaluations,
    })
    return None, diagnostics

  importance = estimate_feature_importance(
      f_batch,
      current_input,
      fully_masked_input,
      current_mask=current_mask,
      feature_groups=feature_groups,
      importance_method=importance_method,
      batch_size=batch_size,
      evaluator=evaluator)

  ranking_scores = np.array(importance.scores, dtype=float)
  ranking = np.array(importance.ranking, dtype=int)
  if ranking_noise_scale > 0 and ranking_scores.size:
    rng = (random_state if isinstance(random_state, np.random.Generator)
           else np.random.default_rng(random_state))
    scale = float(np.std(ranking_scores))
    if scale <= 1e-12:
      scale = max(float(np.max(np.abs(ranking_scores))), 1.0)
    noisy_scores = ranking_scores + rng.normal(
        loc=0.0, scale=ranking_noise_scale * scale, size=ranking_scores.shape)
    ranking = np.argsort(-noisy_scores, kind="mergesort")

  groups = importance.feature_groups
  selected_mask = _empty_mask_like(current_mask)
  selected_group_indices: List[int] = []
  event_indices: List[np.ndarray] = []
  event_values: List[float] = []
  candidate_limit_hit = False

  if max_candidates is None:
    candidate_order = list(ranking)
  else:
    max_candidates = max(1, int(max_candidates))
    candidate_order = list(ranking[:max_candidates])
    if len(ranking) > max_candidates:
      candidate_limit_hit = True

  score = evaluator.evaluate(selected_mask)
  for group_idx in candidate_order:
    selected_mask = np.logical_or(selected_mask, groups[int(group_idx)])
    selected_group_indices.append(int(group_idx))
    score = evaluator.evaluate(selected_mask)
    event_indices.append(_representative_index(groups[int(group_idx)]))
    event_values.append(score)
    if score >= threshold:
      break

  # If the candidate cap was too aggressive, continue with the remaining
  # features so the wrapper still returns a sufficient subset when one exists.
  if score < threshold and max_candidates is not None:
    used = set(candidate_order)
    for group_idx in ranking:
      if int(group_idx) in used:
        continue
      selected_mask = np.logical_or(selected_mask, groups[int(group_idx)])
      selected_group_indices.append(int(group_idx))
      score = evaluator.evaluate(selected_mask)
      event_indices.append(_representative_index(groups[int(group_idx)]))
      event_values.append(score)
      if score >= threshold:
        break

  if score < threshold:
    # Feature groups may not cover the full active mask. Fall back to all active
    # features to preserve SIS sufficiency semantics.
    selected_mask = np.asarray(current_mask, dtype=bool)
    score = evaluator.evaluate(selected_mask)

  if score >= threshold:
    for group_idx in list(reversed(selected_group_indices)):
      group = groups[int(group_idx)]
      trial_mask = np.logical_and(selected_mask, ~group)
      trial_score = evaluator.evaluate(trial_mask)
      event_indices.append(_representative_index(group))
      event_values.append(trial_score)
      if trial_score >= threshold:
        selected_mask = trial_mask
        score = trial_score

  sis_indices = _mask_to_sis_indices(selected_mask)
  result = sis.SISResult(
      sis=sis_indices,
      ordering_over_entire_backselect=_as_ordering_array(
          event_indices, current_mask.ndim),
      values_over_entire_backselect=np.asarray(event_values, dtype=np.float64),
      mask=np.asarray(selected_mask, dtype=bool),
  )

  diagnostics.update({
      "sufficiency_score": float(score),
      "selected_feature_count": int(np.sum(selected_mask)),
      "selected_group_count": int(len(selected_group_indices)),
      "candidate_limit_hit": bool(candidate_limit_hit),
      "importance_scores": ranking_scores.tolist(),
      "ranking": ranking.tolist(),
      "importance_batch_calls": int(importance.batch_calls),
      "importance_individual_evaluations": int(importance.individual_evaluations),
      "model_call_count": int(evaluator.batch_calls + importance.batch_calls),
      "individual_model_evaluations": int(
          evaluator.individual_evaluations + importance.individual_evaluations),
      "threshold_met": bool(score >= threshold),
  })
  return result, diagnostics


def collection_mask(collection: Sequence[sis.SISResult]) -> np.ndarray:
  """Returns the union mask for a SIS collection."""
  if not collection:
    return np.array([], dtype=bool)
  union = np.zeros_like(collection[0].mask, dtype=bool)
  for result in collection:
    union = np.logical_or(union, np.asarray(result.mask, dtype=bool))
  return union


def score_collection(
    f_batch: BatchFunction,
    initial_input: ArrayLike,
    fully_masked_input: ArrayLike,
    collection: Sequence[sis.SISResult],
) -> float:
  """Scores the union of a SIS collection on the original input."""
  if not collection:
    return float("nan")
  mask = collection_mask(collection)
  masked_input = sis.produce_masked_inputs(
      np.asarray(initial_input), np.asarray(fully_masked_input), np.asarray([mask]))
  return float(np.asarray(f_batch(masked_input), dtype=float).reshape(-1)[0])


def shap_guided_sis_collection(
    f_batch: BatchFunction,
    threshold: float,
    initial_input: ArrayLike,
    fully_masked_input: ArrayLike,
    initial_mask: Optional[np.ndarray] = None,
    feature_groups: Optional[Union[np.ndarray, Sequence[Any]]] = None,
    importance_method: str = "perturbation",
    max_candidates: Optional[int] = None,
    batch_size: int = 64,
    return_diagnostics: bool = True,
    baseline_runtime: Optional[float] = None,
    baseline_model_evaluations: Optional[int] = None,
    ranking_noise_scale: float = 0.0,
    random_state: Optional[Union[int, np.random.Generator]] = None,
) -> Union[List[sis.SISResult], Tuple[List[sis.SISResult], Dict[str, Any]]]:
  """Identifies a SIS collection using importance-guided acceleration.

  The return value is compatible with the original ``sis.sis_collection`` when
  ``return_diagnostics=False``. With diagnostics enabled, the function returns
  ``(collection, diagnostics)``.
  """
  start_time = time.perf_counter()
  initial_input = np.asarray(initial_input)
  fully_masked_input = np.asarray(fully_masked_input)
  if initial_mask is None:
    current_mask = sis.make_empty_boolean_mask(initial_input.shape)
  else:
    current_mask = np.asarray(initial_mask, dtype=bool).copy()

  current_input = np.copy(initial_input)
  rng = (random_state if isinstance(random_state, np.random.Generator)
         else np.random.default_rng(random_state))
  collection: List[sis.SISResult] = []
  per_sis_diagnostics: List[Dict[str, Any]] = []
  total_batch_calls = 0
  total_individual_evaluations = 0

  while True:
    seed = int(rng.integers(0, np.iinfo(np.int32).max))
    result, sis_diagnostics = _ranked_find_sis(
        f_batch=f_batch,
        threshold=threshold,
        current_input=current_input,
        current_mask=current_mask,
        fully_masked_input=fully_masked_input,
        feature_groups=feature_groups,
        importance_method=importance_method,
        max_candidates=max_candidates,
        batch_size=batch_size,
        ranking_noise_scale=ranking_noise_scale,
        random_state=seed)
    per_sis_diagnostics.append(sis_diagnostics)
    total_batch_calls += int(sis_diagnostics.get("model_call_count", 0))
    total_individual_evaluations += int(
        sis_diagnostics.get("individual_model_evaluations", 0))

    if result is None:
      break
    if not np.any(result.mask):
      break

    collection.append(result)
    current_input = sis.produce_masked_inputs(
        current_input, fully_masked_input, np.asarray([~result.mask]))[0]
    current_mask = np.logical_and(current_mask, ~result.mask)
    if not np.any(current_mask):
      break

  sis._assert_sis_collection_disjoint(collection)

  runtime = time.perf_counter() - start_time
  first_score = (float(per_sis_diagnostics[0].get("sufficiency_score", np.nan))
                 if per_sis_diagnostics else float("nan"))
  selected_feature_count = int(sum(np.sum(result.mask) for result in collection))
  diagnostics: Dict[str, Any] = {
      "runtime": runtime,
      "model_call_count": int(total_batch_calls),
      "individual_model_evaluations": int(total_individual_evaluations),
      "selected_feature_count": selected_feature_count,
      "subset_size": selected_feature_count,
      "sufficiency_score": first_score,
      "threshold": float(threshold),
      "threshold_met": bool(collection and first_score >= threshold),
      "n_sis": len(collection),
      "importance_method": importance_method,
      "batch_size": int(batch_size),
      "max_candidates": max_candidates,
      "per_sis": per_sis_diagnostics,
      "baseline_runtime": baseline_runtime,
      "baseline_model_evaluations": baseline_model_evaluations,
  }
  if baseline_runtime is not None and baseline_runtime > 0:
    diagnostics["estimated_runtime_speedup_pct"] = (
        (baseline_runtime - runtime) / baseline_runtime * 100.0)
  else:
    diagnostics["estimated_runtime_speedup_pct"] = None
  if baseline_model_evaluations is not None and baseline_model_evaluations > 0:
    diagnostics["estimated_evaluation_reduction_pct"] = (
        (baseline_model_evaluations - total_individual_evaluations) /
        baseline_model_evaluations * 100.0)
  else:
    diagnostics["estimated_evaluation_reduction_pct"] = None

  if return_diagnostics:
    return collection, diagnostics
  return collection

