# coding=utf-8
"""Benchmarks original SIS against extension variants.

This benchmark intentionally uses a small deterministic synthetic image model so
it can run on a normal laptop without network access or large dependencies.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from sufficient_input_subsets import sis
from sufficient_input_subsets.hierarchical_sis import hierarchical_sis_collection
from sufficient_input_subsets.probabilistic_sis import probabilistic_sis_collection
from sufficient_input_subsets.shap_guided_sis import (
    collection_mask,
    shap_guided_sis_collection,
)
from sufficient_input_subsets.stability_metrics import (
    explanation_stability,
    extract_explanation_mask,
)


class CountingFunction:
  """Counts batched calls and individual examples evaluated by f_batch."""

  def __init__(self, f_batch: Callable[[np.ndarray], np.ndarray]):
    self.f_batch = f_batch
    self.batch_calls = 0
    self.individual_evaluations = 0

  def __call__(self, batch: np.ndarray) -> np.ndarray:
    batch = np.asarray(batch)
    self.batch_calls += 1
    self.individual_evaluations += batch.shape[0]
    return np.asarray(self.f_batch(batch), dtype=float).reshape(-1)


def _sigmoid(values: np.ndarray) -> np.ndarray:
  return 1.0 / (1.0 + np.exp(-values))


def make_synthetic_image_problem(
    image_size: int = 8,
) -> Tuple[np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray]]:
  """Returns an image, masked baseline, and confidence model."""
  image = np.zeros((image_size, image_size), dtype=float)
  weights = np.zeros_like(image)
  image[1:3, 1:3] = 1.0
  image[5:7, 5:7] = 0.9
  image[3, 4] = 0.6
  image[4, 3] = 0.6
  weights[1:3, 1:3] = 2.2
  weights[5:7, 5:7] = 1.25
  weights[3, 4] = 0.8
  weights[4, 3] = 0.8
  fully_masked = np.zeros_like(image)

  def f_batch(batch: np.ndarray) -> np.ndarray:
    batch = np.asarray(batch, dtype=float)
    logits = np.sum(batch * weights, axis=(1, 2)) - 4.0
    return _sigmoid(logits)

  return image, fully_masked, f_batch


def _collection_summary(
    f_batch: Callable[[np.ndarray], np.ndarray],
    initial_input: np.ndarray,
    fully_masked_input: np.ndarray,
    collection,
    threshold: float,
) -> Dict[str, Any]:
  if not collection:
    return {
        "subset_size": 0,
        "sufficiency_score": float("nan"),
        "threshold_met": False,
        "n_sis": 0,
    }
  mask = collection_mask(collection)
  masked_input = sis.produce_masked_inputs(
      initial_input, fully_masked_input, np.asarray([mask]))
  score = float(np.asarray(f_batch(masked_input)).reshape(-1)[0])
  return {
      "subset_size": int(np.sum(mask)),
      "sufficiency_score": score,
      "threshold_met": bool(score >= threshold),
      "n_sis": len(collection),
  }


def _json_safe(value: Any) -> Any:
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, (np.floating, np.integer)):
    return value.item()
  if isinstance(value, dict):
    return {str(k): _json_safe(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_json_safe(v) for v in value]
  return value


def _stability_probe(
    method: Callable[[Callable[[np.ndarray], np.ndarray], float, np.ndarray, np.ndarray], Any],
    f_batch: Callable[[np.ndarray], np.ndarray],
    threshold: float,
    initial_input: np.ndarray,
    fully_masked_input: np.ndarray,
) -> float:
  masks = []
  rng = np.random.default_rng(123)
  for scale in (0.0, 0.01, 0.02):
    perturbed = np.clip(initial_input + rng.normal(0.0, scale, initial_input.shape), 0.0, 1.0)
    result = method(f_batch, threshold, perturbed, fully_masked_input)
    mask = extract_explanation_mask(result)
    if mask.size:
      masks.append(mask)
  if not masks:
    return 0.0
  return float(explanation_stability(masks, metric="iou")["mean_pairwise"])


def run_benchmark(
    threshold: float = 0.75,
    image_size: int = 8,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
  initial_input, fully_masked_input, base_f = make_synthetic_image_problem(image_size)
  output_dir = output_dir or Path(__file__).resolve().parent / "results"
  output_dir.mkdir(parents=True, exist_ok=True)
  methods: Dict[str, Dict[str, Any]] = {}

  counted = CountingFunction(base_f)
  start = time.perf_counter()
  baseline_collection = sis.sis_collection(
      counted, threshold, initial_input, fully_masked_input)
  baseline_runtime = time.perf_counter() - start
  methods["original_sis"] = {
      "runtime": baseline_runtime,
      "model_call_count": counted.batch_calls,
      "individual_model_evaluations": counted.individual_evaluations,
      **_collection_summary(base_f, initial_input, fully_masked_input,
                            baseline_collection, threshold),
  }

  counted = CountingFunction(base_f)
  shap_collection, shap_diagnostics = shap_guided_sis_collection(
      counted,
      threshold,
      initial_input,
      fully_masked_input,
      batch_size=64,
      baseline_runtime=baseline_runtime,
      baseline_model_evaluations=methods["original_sis"]["individual_model_evaluations"],
      return_diagnostics=True)
  methods["shap_guided_sis"] = {
      "runtime": shap_diagnostics["runtime"],
      "model_call_count": counted.batch_calls,
      "individual_model_evaluations": counted.individual_evaluations,
      **_collection_summary(base_f, initial_input, fully_masked_input,
                            shap_collection, threshold),
      "diagnostics": shap_diagnostics,
  }

  counted = CountingFunction(base_f)
  start = time.perf_counter()
  probabilistic = probabilistic_sis_collection(
      counted,
      threshold,
      initial_input,
      fully_masked_input,
      n_samples=5,
      noise_scale=0.08,
      random_state=7)
  prob_runtime = time.perf_counter() - start
  prob_mask = probabilistic.inclusion_probabilities >= 0.5
  prob_score = float(base_f(sis.produce_masked_inputs(
      initial_input, fully_masked_input, np.asarray([prob_mask])))[0])
  methods["probabilistic_sis"] = {
      "runtime": prob_runtime,
      "model_call_count": counted.batch_calls,
      "individual_model_evaluations": counted.individual_evaluations,
      "subset_size": int(np.sum(prob_mask)),
      "sufficiency_score": prob_score,
      "threshold_met": bool(prob_score >= threshold),
      "n_sis": len(probabilistic.sampled_results),
      "mean_subset_size": probabilistic.mean_subset_size,
      "variance_subset_size": probabilistic.variance_subset_size,
      "stability_summary": probabilistic.stability_summary,
  }

  counted = CountingFunction(base_f)
  start = time.perf_counter()
  hierarchical = hierarchical_sis_collection(
      counted,
      threshold,
      initial_input,
      fully_masked_input,
      levels=(2, 4, image_size),
      mode="pixel")
  hier_runtime = time.perf_counter() - start
  hier_mask = extract_explanation_mask(hierarchical)
  hier_score = float(base_f(sis.produce_masked_inputs(
      initial_input, fully_masked_input, np.asarray([hier_mask])))[0])
  methods["hierarchical_sis"] = {
      "runtime": hier_runtime,
      "model_call_count": counted.batch_calls,
      "individual_model_evaluations": counted.individual_evaluations,
      "subset_size": int(np.sum(hier_mask)),
      "sufficiency_score": hier_score,
      "threshold_met": bool(hier_score >= threshold),
      "n_sis": len(hierarchical.final_masks),
      "diagnostics": hierarchical.diagnostics,
  }

  stability_methods = {
      "original_sis": lambda f, t, x, m: sis.sis_collection(f, t, x, m),
      "shap_guided_sis": lambda f, t, x, m: shap_guided_sis_collection(
          f, t, x, m, return_diagnostics=False),
      "probabilistic_sis": lambda f, t, x, m: probabilistic_sis_collection(
          f, t, x, m, n_samples=3, noise_scale=0.08, random_state=11),
      "hierarchical_sis": lambda f, t, x, m: hierarchical_sis_collection(
          f, t, x, m, levels=(2, 4, image_size), mode="pixel"),
  }
  for name, method in stability_methods.items():
    methods[name]["stability_score"] = _stability_probe(
        method, base_f, threshold, initial_input, fully_masked_input)

  baseline_evals = methods["original_sis"]["individual_model_evaluations"]
  for name, row in methods.items():
    row["speedup_percentage_against_baseline"] = (
        (baseline_evals - row["individual_model_evaluations"]) /
        baseline_evals * 100.0 if baseline_evals else 0.0)

  shap_reduction = methods["shap_guided_sis"]["speedup_percentage_against_baseline"]
  achieved_20 = shap_reduction >= 20.0
  report = {
      "problem": {
          "type": "synthetic_image_classifier",
          "image_size": image_size,
          "threshold": threshold,
          "initial_confidence": float(base_f(np.asarray([initial_input]))[0]),
          "fully_masked_confidence": float(base_f(np.asarray([fully_masked_input]))[0]),
      },
      "methods": methods,
      "shap_guided_overhead_reduction_pct": shap_reduction,
      "shap_guided_achieved_around_20pct_reduction": bool(achieved_20),
      "interpretation": (
          "SHAP-guided SIS achieved at least a 20% individual-evaluation "
          "reduction on this benchmark."
          if achieved_20 else
          "SHAP-guided SIS did not reach a 20% individual-evaluation reduction "
          "on this benchmark; report the measured value and tune candidate "
          "limits, grouping, and cache reuse next."),
  }

  report_path = output_dir / "benchmark_report.json"
  summary_path = output_dir / "benchmark_summary.md"
  report_path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
  summary_path.write_text(_make_summary(report), encoding="utf-8")
  return report


def _make_summary(report: Dict[str, Any]) -> str:
  methods = report["methods"]
  lines = [
      "# SIS Extension Benchmark Summary",
      "",
      "Synthetic image benchmark with a deterministic confidence function.",
      "",
      "| Method | Runtime (s) | f_batch calls | Individual evals | Subset size | Score | Met threshold | Speedup vs baseline | Stability |",
      "| --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: |",
  ]
  for name, row in methods.items():
    lines.append(
        "| {name} | {runtime:.4f} | {calls} | {evals} | {subset} | {score:.4f} | {met} | {speedup:.2f}% | {stability:.3f} |".format(
            name=name,
            runtime=float(row["runtime"]),
            calls=row["model_call_count"],
            evals=row["individual_model_evaluations"],
            subset=row["subset_size"],
            score=float(row["sufficiency_score"]),
            met="yes" if row["threshold_met"] else "no",
            speedup=float(row["speedup_percentage_against_baseline"]),
            stability=float(row.get("stability_score", 0.0)),
        ))
  lines.extend([
      "",
      "## SHAP-guided overhead reduction",
      "",
      "Measured reduction in individual model evaluations: **{:.2f}%**.".format(
          float(report["shap_guided_overhead_reduction_pct"])),
      "",
      report["interpretation"],
      "",
      "Next optimization hooks if the measured reduction is below target: reduce "
      "`max_candidates`, use larger image-region feature groups before pixel "
      "refinement, reuse importance caches across nearby perturbations, and batch "
      "larger candidate sets when the model supports it.",
      "",
  ])
  return "\n".join(lines)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--threshold", type=float, default=0.75)
  parser.add_argument("--image_size", type=int, default=8)
  parser.add_argument(
      "--output_dir",
      type=Path,
      default=Path(__file__).resolve().parent / "results")
  args = parser.parse_args()
  report = run_benchmark(
      threshold=args.threshold,
      image_size=args.image_size,
      output_dir=args.output_dir)
  print(_make_summary(report))


if __name__ == "__main__":
  main()

