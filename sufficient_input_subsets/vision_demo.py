# coding=utf-8
"""Small runnable vision demo for SIS extensions.

The demo prefers the lightweight scikit-learn digits dataset and falls back to a
synthetic image classifier when scikit-learn is unavailable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from sufficient_input_subsets import sis
from sufficient_input_subsets.benchmark_sis import make_synthetic_image_problem
from sufficient_input_subsets.hierarchical_sis import (
    hierarchical_sis_collection,
    plot_hierarchical_masks,
)
from sufficient_input_subsets.probabilistic_sis import (
    plot_inclusion_heatmap,
    probabilistic_sis_collection,
)
from sufficient_input_subsets.shap_guided_sis import (
    collection_mask,
    shap_guided_sis_collection,
)
from sufficient_input_subsets.stability_metrics import perturbation_stability


def _load_digits_problem(random_state: int = 0):
  try:
    from sklearn.datasets import load_digits
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
  except Exception:
    return None

  digits = load_digits()
  images = digits.images.astype(float) / 16.0
  labels = digits.target
  x_train, x_test, y_train, y_test = train_test_split(
      images, labels, test_size=0.25, random_state=random_state, stratify=labels)
  model = LogisticRegression(max_iter=500, solver="lbfgs", multi_class="auto")
  model.fit(x_train.reshape((x_train.shape[0], -1)), y_train)
  probabilities = model.predict_proba(x_test.reshape((x_test.shape[0], -1)))
  predictions = np.argmax(probabilities, axis=1)
  correct = np.where(predictions == y_test)[0]
  if correct.size == 0:
    return None
  confidences = probabilities[correct, predictions[correct]]
  chosen = int(correct[np.argmax(confidences)])
  image = x_test[chosen]
  target = int(predictions[chosen])
  confidence = float(probabilities[chosen, target])
  fully_masked = np.zeros_like(image)

  def f_batch(batch: np.ndarray) -> np.ndarray:
    batch = np.asarray(batch, dtype=float)
    probs = model.predict_proba(batch.reshape((batch.shape[0], -1)))
    return probs[:, target]

  return {
      "dataset": "sklearn_digits",
      "image": image,
      "fully_masked": fully_masked,
      "f_batch": f_batch,
      "target": target,
      "confidence": confidence,
  }


def load_vision_problem(random_state: int = 0) -> Dict[str, Any]:
  """Loads digits or a synthetic fallback problem."""
  digits = _load_digits_problem(random_state=random_state)
  if digits is not None:
    return digits
  image, fully_masked, f_batch = make_synthetic_image_problem(image_size=8)
  confidence = float(f_batch(np.asarray([image]))[0])
  return {
      "dataset": "synthetic_image",
      "image": image,
      "fully_masked": fully_masked,
      "f_batch": f_batch,
      "target": 1,
      "confidence": confidence,
  }


def _mask_or_empty(collection, shape: Tuple[int, ...]) -> np.ndarray:
  if collection:
    return collection_mask(collection)
  return np.zeros(shape, dtype=bool)


def _display_image(array: np.ndarray) -> np.ndarray:
  image = np.asarray(array)
  if image.ndim == 1:
    image = np.expand_dims(image, axis=0)
  elif image.ndim > 2:
    image = np.mean(image, axis=-1)
  return image


def _save_single_image(path: Path, image: np.ndarray, title: str, cmap: str = "gray") -> None:
  import matplotlib.pyplot as plt

  fig, ax = plt.subplots(figsize=(3, 3))
  ax.imshow(_display_image(image), cmap=cmap)
  ax.set_title(title)
  ax.set_xticks([])
  ax.set_yticks([])
  fig.tight_layout()
  fig.savefig(path, dpi=160)
  plt.close(fig)


def run_demo(
    output_dir: Optional[Path] = None,
    threshold: float | None = None,
    random_state: int = 0,
    n_probabilistic_samples: int = 10,
) -> Dict[str, Any]:
  """Runs the demo and saves visualizations."""
  try:
    import matplotlib.pyplot as plt
  except Exception as exc:
    raise ImportError("matplotlib is required to save demo visualizations.") from exc

  output_dir = output_dir or Path(__file__).resolve().parent / "results"
  output_dir.mkdir(parents=True, exist_ok=True)

  problem = load_vision_problem(random_state=random_state)
  image = problem["image"]
  fully_masked = problem["fully_masked"]
  f_batch = problem["f_batch"]
  confidence = float(problem["confidence"])
  threshold = float(threshold if threshold is not None else max(0.5, min(0.9, 0.85 * confidence)))

  baseline_collection = sis.sis_collection(f_batch, threshold, image, fully_masked)
  shap_collection, shap_diagnostics = shap_guided_sis_collection(
      f_batch, threshold, image, fully_masked, return_diagnostics=True)
  probabilistic = probabilistic_sis_collection(
      f_batch,
      threshold,
      image,
      fully_masked,
      n_samples=n_probabilistic_samples,
      noise_scale=0.08,
      random_state=random_state)
  hierarchical = hierarchical_sis_collection(
      f_batch, threshold, image, fully_masked, levels=(2, 4, image.shape[0]), mode="pixel")

  baseline_mask = _mask_or_empty(baseline_collection, image.shape)
  shap_mask = _mask_or_empty(shap_collection, image.shape)
  hierarchical_mask = hierarchical.level_masks[-1] if hierarchical.level_masks else np.zeros_like(image, dtype=bool)

  rng = np.random.default_rng(random_state)

  def perturb_fn(x: np.ndarray) -> np.ndarray:
    return np.clip(x + rng.normal(0.0, 0.02, x.shape), 0.0, 1.0)

  stability = perturbation_stability(
      f_batch,
      lambda f, t, x, m: shap_guided_sis_collection(f, t, x, m, return_diagnostics=False),
      np.asarray([image]),
      fully_masked,
      perturb_fn,
      threshold,
      n_perturbations=5)

  _save_single_image(output_dir / "vision_demo_original.png", image, "Original")
  _save_single_image(output_dir / "vision_demo_baseline_sis_mask.png", baseline_mask, "Baseline SIS", cmap="viridis")
  _save_single_image(output_dir / "vision_demo_shap_guided_sis_mask.png", shap_mask, "SHAP-guided SIS", cmap="viridis")

  fig, ax = plt.subplots(figsize=(3.4, 3))
  plot_inclusion_heatmap(probabilistic.inclusion_probabilities, ax=ax, title="Inclusion probability")
  fig.tight_layout()
  fig.savefig(output_dir / "vision_demo_probabilistic_heatmap.png", dpi=160)
  plt.close(fig)

  fig, axes = plt.subplots(1, len(hierarchical.level_masks), figsize=(3 * len(hierarchical.level_masks), 3))
  plot_hierarchical_masks(hierarchical.level_masks, axes=axes)
  fig.tight_layout()
  fig.savefig(output_dir / "vision_demo_hierarchical_regions.png", dpi=160)
  plt.close(fig)

  fig, ax = plt.subplots(figsize=(4, 3))
  labels = ["drift", "retention"]
  values = [
      float(stability["mean_explanation_drift"]),
      float(stability["mean_confidence_retention"])
      if np.isfinite(stability["mean_confidence_retention"]) else 0.0,
  ]
  ax.bar(labels, values, color=["#4c78a8", "#f58518"])
  ax.set_ylim(0.0, max(1.0, max(values) * 1.1))
  ax.set_title("Perturbation stability")
  fig.tight_layout()
  fig.savefig(output_dir / "vision_demo_perturbation_stability.png", dpi=160)
  plt.close(fig)

  summary = {
      "dataset": problem["dataset"],
      "target": int(problem["target"]),
      "initial_confidence": confidence,
      "threshold": threshold,
      "baseline_subset_size": int(np.sum(baseline_mask)),
      "shap_guided_subset_size": int(np.sum(shap_mask)),
      "probabilistic_mean_subset_size": probabilistic.mean_subset_size,
      "hierarchical_final_subset_size": int(np.sum(hierarchical_mask)),
      "shap_guided_diagnostics": shap_diagnostics,
      "probabilistic_confidence_summary": probabilistic.confidence_summary,
      "hierarchical_diagnostics": hierarchical.diagnostics,
      "stability": {
          "mean_explanation_drift": stability["mean_explanation_drift"],
          "mean_confidence_retention": stability["mean_confidence_retention"],
      },
      "artifacts": {
          "original": str(output_dir / "vision_demo_original.png"),
          "baseline_sis_mask": str(output_dir / "vision_demo_baseline_sis_mask.png"),
          "shap_guided_sis_mask": str(output_dir / "vision_demo_shap_guided_sis_mask.png"),
          "probabilistic_heatmap": str(output_dir / "vision_demo_probabilistic_heatmap.png"),
          "hierarchical_regions": str(output_dir / "vision_demo_hierarchical_regions.png"),
          "perturbation_stability": str(output_dir / "vision_demo_perturbation_stability.png"),
      },
  }
  (output_dir / "vision_demo_summary.json").write_text(
      json.dumps(summary, indent=2), encoding="utf-8")
  return summary


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--output_dir",
      type=Path,
      default=Path(__file__).resolve().parent / "results")
  parser.add_argument("--threshold", type=float, default=None)
  parser.add_argument("--random_state", type=int, default=0)
  parser.add_argument("--n_probabilistic_samples", type=int, default=10)
  args = parser.parse_args()
  summary = run_demo(
      output_dir=args.output_dir,
      threshold=args.threshold,
      random_state=args.random_state,
      n_probabilistic_samples=args.n_probabilistic_samples)
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()

