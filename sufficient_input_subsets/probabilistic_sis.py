# coding=utf-8
"""Probabilistic Sufficient Input Subsets.

Probabilistic SIS samples multiple plausible explanations by injecting
controlled noise into the feature-ranking stage. The resulting inclusion
probability map is useful when a single deterministic SIS is too brittle or
when multiple near-equivalent rationales exist.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from sufficient_input_subsets import sis
from sufficient_input_subsets.shap_guided_sis import (
    collection_mask,
    shap_guided_sis_collection,
)
from sufficient_input_subsets.stability_metrics import explanation_stability


@dataclasses.dataclass
class ProbabilisticSISResult:
  """Container returned by ``probabilistic_sis_collection``."""

  sampled_results: List[List[sis.SISResult]]
  inclusion_probabilities: np.ndarray
  mean_subset_size: float
  variance_subset_size: float
  confidence_summary: Dict[str, Any]
  stability_summary: Dict[str, Any]
  diagnostics: Dict[str, Any]


def _union_or_empty(
    collection: Sequence[sis.SISResult],
    mask_shape: Sequence[int],
) -> np.ndarray:
  if collection:
    return collection_mask(collection)
  return np.zeros(tuple(mask_shape), dtype=bool)


def probabilistic_sis_collection(
    f_batch,
    threshold,
    initial_input,
    fully_masked_input,
    n_samples=30,
    noise_scale=0.05,
    feature_groups=None,
    random_state=0,
    use_shap_guidance=True,
):
  """Samples plausible SIS explanations and estimates inclusion probabilities.

  Args:
    f_batch: Batched black-box function returning one scalar per input.
    threshold: Sufficiency threshold.
    initial_input: Input to explain.
    fully_masked_input: Baseline/masked input.
    n_samples: Number of stochastic SIS samples.
    noise_scale: Standard deviation multiplier applied to ranking scores.
    feature_groups: Optional feature/region groups.
    random_state: Seed for reproducible sampling.
    use_shap_guidance: If True, use the SHAP-inspired ranking wrapper. If False,
      the original SIS collection is repeated deterministically.

  Returns:
    ``ProbabilisticSISResult`` with sampled collections and probability maps.
  """
  initial_input = np.asarray(initial_input)
  fully_masked_input = np.asarray(fully_masked_input)
  rng = np.random.default_rng(random_state)
  n_samples = int(n_samples)
  if n_samples <= 0:
    raise ValueError("n_samples must be positive.")

  sampled_results: List[List[sis.SISResult]] = []
  masks: List[np.ndarray] = []
  subset_sizes: List[int] = []
  sufficiency_scores: List[float] = []
  diagnostics_per_sample: List[Dict[str, Any]] = []
  mask_shape = initial_input.shape

  for _ in range(n_samples):
    if use_shap_guidance:
      sample_seed = int(rng.integers(0, np.iinfo(np.int32).max))
      collection, diagnostics = shap_guided_sis_collection(
          f_batch,
          threshold,
          initial_input,
          fully_masked_input,
          feature_groups=feature_groups,
          importance_method="perturbation",
          ranking_noise_scale=float(noise_scale),
          random_state=sample_seed,
          return_diagnostics=True)
    else:
      collection = sis.sis_collection(
          f_batch, threshold, initial_input, fully_masked_input)
      diagnostics = {"runtime": None, "model_call_count": None}

    sampled_results.append(collection)
    mask = _union_or_empty(collection, mask_shape)
    masks.append(mask)
    subset_sizes.append(int(np.sum(mask)))
    if collection:
      masked_input = sis.produce_masked_inputs(
          initial_input, fully_masked_input, np.asarray([mask]))
      sufficiency_scores.append(
          float(np.asarray(f_batch(masked_input)).reshape(-1)[0]))
    diagnostics_per_sample.append(diagnostics)

  inclusion_probabilities = np.mean(np.asarray(masks, dtype=float), axis=0)
  stability_summary = explanation_stability(masks, metric="iou")
  confidence_summary = {
      "mean_sufficiency_score": (
          float(np.mean(sufficiency_scores)) if sufficiency_scores else float("nan")),
      "min_sufficiency_score": (
          float(np.min(sufficiency_scores)) if sufficiency_scores else float("nan")),
      "threshold": float(threshold),
      "threshold_met_rate": float(
          np.mean(np.asarray(sufficiency_scores) >= threshold))
      if sufficiency_scores else 0.0,
      "mean_inclusion_probability": float(np.mean(inclusion_probabilities)),
      "max_inclusion_probability": float(np.max(inclusion_probabilities)),
  }
  diagnostics = {
      "n_samples": n_samples,
      "noise_scale": float(noise_scale),
      "random_state": random_state,
      "use_shap_guidance": bool(use_shap_guidance),
      "per_sample": diagnostics_per_sample,
      "total_model_call_count": int(sum(
          d.get("model_call_count", 0) or 0 for d in diagnostics_per_sample)),
      "total_individual_model_evaluations": int(sum(
          d.get("individual_model_evaluations", 0) or 0
          for d in diagnostics_per_sample)),
  }

  return ProbabilisticSISResult(
      sampled_results=sampled_results,
      inclusion_probabilities=inclusion_probabilities,
      mean_subset_size=float(np.mean(subset_sizes)),
      variance_subset_size=float(np.var(subset_sizes)),
      confidence_summary=confidence_summary,
      stability_summary=stability_summary,
      diagnostics=diagnostics,
  )


def plot_inclusion_heatmap(
    inclusion_probabilities: np.ndarray,
    ax=None,
    title: str = "Probabilistic SIS inclusion probability",
):
  """Plots an inclusion-probability heatmap for image-like inputs."""
  try:
    import matplotlib.pyplot as plt
  except Exception as exc:
    raise ImportError("matplotlib is required for plotting.") from exc

  probabilities = np.asarray(inclusion_probabilities, dtype=float)
  if probabilities.ndim == 1:
    image = np.expand_dims(probabilities, axis=0)
  elif probabilities.ndim == 2:
    image = probabilities
  else:
    image = np.mean(probabilities, axis=-1)

  if ax is None:
    _, ax = plt.subplots(figsize=(4, 3))
  im = ax.imshow(image, cmap="magma", vmin=0.0, vmax=1.0)
  ax.set_title(title)
  ax.set_xticks([])
  ax.set_yticks([])
  plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
  return ax


def top_k_stable_features(
    inclusion_probabilities: np.ndarray,
    k: int = 10,
) -> Union[List[Dict[str, Any]], Any]:
  """Returns the top-k most consistently included features or regions."""
  probabilities = np.asarray(inclusion_probabilities, dtype=float)
  coords = np.transpose(np.nonzero(np.ones(probabilities.shape, dtype=bool)))
  flat_probs = probabilities.reshape(-1)
  order = np.argsort(-flat_probs, kind="mergesort")[:int(k)]
  rows = [
      {"rank": rank + 1, "index": tuple(coords[idx]), "probability": float(flat_probs[idx])}
      for rank, idx in enumerate(order)
  ]
  try:
    import pandas as pd  # type: ignore
  except Exception:
    return rows
  return pd.DataFrame(rows)
