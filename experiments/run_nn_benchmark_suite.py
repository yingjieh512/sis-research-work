# coding=utf-8
"""Run SIS neural-network benchmarks over lightweight datasets.

The suite runs the same experiment harness for each requested dataset and
writes an aggregate report. MNIST is included as an optional benchmark path:
it uses TorchVision when available and records an unavailable/failed status
instead of fabricating results when dependencies or downloads are missing.

Example:

  python -m experiments.run_nn_benchmark_suite --datasets digits,mnist
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from experiments import run_nn_sis_experiments as harness


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "results" / "sis_nn_benchmarks"


def _json_safe(value: Any) -> Any:
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, (np.floating, np.integer)):
    return value.item()
  if isinstance(value, float) and math.isnan(value):
    return None
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, dict):
    return {str(k): _json_safe(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_json_safe(v) for v in value]
  return value


def parse_dataset_list(raw: str) -> List[str]:
  """Parses a comma-separated dataset list."""
  datasets = [item.strip().lower() for item in raw.split(",") if item.strip()]
  if not datasets:
    raise ValueError("--datasets must name at least one dataset.")
  supported = {"digits", "mnist", "fashion_mnist", "cifar10"}
  unsupported = sorted(set(datasets) - supported)
  if unsupported:
    raise ValueError("Unsupported datasets: %s" % ", ".join(unsupported))
  return datasets


def _append_flag(flag_args: List[str], flag: str, value: Optional[Any]) -> None:
  if value is not None:
    flag_args.extend([flag, str(value)])


def make_harness_argv(dataset: str, args: argparse.Namespace, output_dir: Path) -> List[str]:
  """Builds arguments for one dataset run in the benchmark suite."""
  argv = [
      "--dataset", dataset,
      "--model", "mlp",
      "--max-examples", str(args.max_examples),
      "--seed", str(args.seed),
      "--threshold-mode", args.threshold_mode,
      "--threshold", str(args.threshold),
      "--relative-fraction", str(args.relative_fraction),
      "--min-threshold", str(args.min_threshold),
      "--min-confidence", str(args.min_confidence),
      "--output-dir", str(output_dir),
      "--max-iter", str(args.max_iter),
      "--batch-size", str(args.batch_size),
      "--probabilistic-samples", str(args.probabilistic_samples),
      "--probabilistic-noise", str(args.probabilistic_noise),
      "--probabilistic-mask-threshold", str(args.probabilistic_mask_threshold),
      "--stability-perturbations", str(args.stability_perturbations),
      "--stability-noise", str(args.stability_noise),
  ]
  _append_flag(argv, "--max-candidates", args.max_candidates)

  if args.skip_probabilistic:
    argv.append("--skip-probabilistic")
  if args.skip_hierarchical:
    argv.append("--skip-hierarchical")

  if dataset == "digits":
    argv.extend(["--hierarchy-levels", args.digits_hierarchy_levels])
  else:
    argv.extend([
        "--train-subset", str(args.mnist_train_subset),
        "--test-subset", str(args.mnist_test_subset),
        "--hierarchy-levels", args.mnist_hierarchy_levels,
    ])
    _append_flag(argv, "--image-size", args.mnist_image_size)

  return argv


def _load_json(path: str) -> Dict[str, Any]:
  return json.loads(Path(path).read_text(encoding="utf-8"))


def _method_summary(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
  methods = sorted({row["method"] for row in rows})
  summaries = []
  for method in methods:
    method_rows = [row for row in rows if row["method"] == method]
    summaries.append({
        "method": method,
        "threshold_satisfied_rate": float(np.mean([
            bool(row.get("threshold_satisfied", False)) for row in method_rows])),
        "mean_final_confidence": float(np.mean([
            float(row.get("final_confidence", np.nan)) for row in method_rows])),
        "mean_subset_size": float(np.mean([
            float(row.get("subset_size", np.nan)) for row in method_rows])),
        "mean_individual_model_evaluations": float(np.mean([
            float(row.get("individual_model_evaluations", np.nan))
            for row in method_rows])),
        "mean_batched_function_calls": float(np.mean([
            float(row.get("batched_function_calls", np.nan)) for row in method_rows])),
        "mean_evaluation_reduction_vs_baseline_pct": float(np.mean([
            float(row.get("evaluation_reduction_vs_baseline_pct", np.nan))
            for row in method_rows])),
        "mean_runtime_sec": float(np.mean([
            float(row.get("wall_clock_runtime_sec", np.nan)) for row in method_rows])),
    })
  return summaries


def _write_aggregate_csv(records: Sequence[Dict[str, Any]], output_dir: Path) -> Path:
  csv_path = output_dir / "benchmark_suite_results.csv"
  rows = []
  for record in records:
    if record["status"] != "completed":
      rows.append({
          "dataset": record["dataset"],
          "status": record["status"],
          "method": "",
          "threshold_satisfied_rate": "",
          "mean_final_confidence": "",
          "mean_subset_size": "",
          "mean_individual_model_evaluations": "",
          "mean_batched_function_calls": "",
          "mean_evaluation_reduction_vs_baseline_pct": "",
          "mean_runtime_sec": "",
          "notes": record.get("error", ""),
      })
      continue
    for summary in record["method_summary"]:
      row = {"dataset": record["dataset"], "status": "completed", **summary, "notes": ""}
      rows.append(row)

  fieldnames = [
      "dataset",
      "status",
      "method",
      "threshold_satisfied_rate",
      "mean_final_confidence",
      "mean_subset_size",
      "mean_individual_model_evaluations",
      "mean_batched_function_calls",
      "mean_evaluation_reduction_vs_baseline_pct",
      "mean_runtime_sec",
      "notes",
  ]
  with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
      writer.writerow(row)
  return csv_path


def _write_markdown_summary(records: Sequence[Dict[str, Any]], output_dir: Path) -> Path:
  summary_path = output_dir / "benchmark_suite_summary.md"
  lines = [
      "# SIS Neural-Network Benchmark Suite",
      "",
      "This report is generated from local benchmark runs. A dataset is marked "
      "`unavailable` or `failed` when dependencies, downloads, or runtime "
      "conditions prevent the benchmark from completing.",
      "",
      "| Dataset | Status | Notes |",
      "| --- | --- | --- |",
  ]
  for record in records:
    notes = record.get("error", "")
    if record["status"] == "completed":
      metadata = record.get("metadata", {})
      notes = (
          "accuracy %.4f; output `%s`" %
          (float(metadata.get("test_accuracy", float("nan"))), record.get("output_dir", "")))
    lines.append("| %s | %s | %s |" % (
        record["dataset"], record["status"], str(notes).replace("|", "\\|")))

  for record in records:
    if record["status"] != "completed":
      continue
    lines.extend([
        "",
        "## %s" % record["dataset"],
        "",
        "| Method | Threshold rate | Mean confidence | Mean subset | Mean evals | Eval reduction | Mean runtime (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for summary in record["method_summary"]:
      lines.append(
          "| {method} | {threshold_satisfied_rate:.3f} | {mean_final_confidence:.6f} | "
          "{mean_subset_size:.2f} | {mean_individual_model_evaluations:.1f} | "
          "{mean_evaluation_reduction_vs_baseline_pct:.2f}% | {mean_runtime_sec:.4f} |"
          .format(**summary))

  lines.extend([
      "",
      "## Honesty Note",
      "",
      "Only rows with status `completed` are measured benchmark evidence. "
      "MNIST should be discussed as measured only when its row is completed "
      "and the referenced per-dataset result files are present.",
  ])
  summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
  return summary_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--datasets", default="digits,mnist")
  parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
  parser.add_argument("--max-examples", type=int, default=1)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--threshold-mode", choices=["fixed", "relative"], default="relative")
  parser.add_argument("--threshold", type=float, default=0.8)
  parser.add_argument("--relative-fraction", type=float, default=0.85)
  parser.add_argument("--min-threshold", type=float, default=0.5)
  parser.add_argument("--min-confidence", type=float, default=0.70)
  parser.add_argument("--max-iter", type=int, default=120)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--max-candidates", type=int, default=None)
  parser.add_argument("--probabilistic-samples", type=int, default=3)
  parser.add_argument("--probabilistic-noise", type=float, default=0.05)
  parser.add_argument("--probabilistic-mask-threshold", type=float, default=0.5)
  parser.add_argument("--stability-perturbations", type=int, default=0)
  parser.add_argument("--stability-noise", type=float, default=0.02)
  parser.add_argument("--skip-probabilistic", action="store_true")
  parser.add_argument("--skip-hierarchical", action="store_true")
  parser.add_argument("--digits-hierarchy-levels", default="2,4,8")
  parser.add_argument("--mnist-train-subset", type=int, default=2000)
  parser.add_argument("--mnist-test-subset", type=int, default=500)
  parser.add_argument("--mnist-image-size", type=int, default=14)
  parser.add_argument("--mnist-hierarchy-levels", default="2,7,14")
  parser.add_argument(
      "--fail-on-error",
      action="store_true",
      help="Raise dataset errors instead of recording them in the report.")
  return parser.parse_args(argv)


def run_suite(args: argparse.Namespace) -> Dict[str, Any]:
  stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  base_output_dir = Path(args.output_dir)
  if not base_output_dir.is_absolute():
    base_output_dir = ROOT / base_output_dir
  output_dir = base_output_dir / ("benchmark_suite_%s" % stamp)
  output_dir.mkdir(parents=True, exist_ok=True)

  records: List[Dict[str, Any]] = []
  for dataset in parse_dataset_list(args.datasets):
    dataset_output_dir = output_dir / ("%s_mlp" % dataset)
    argv = make_harness_argv(dataset, args, dataset_output_dir)
    try:
      harness_args = harness.parse_args(argv)
      summary = harness.run_experiment(harness_args)
      payload = _load_json(summary["json_path"])
      rows = payload["results"]
      records.append({
          "dataset": dataset,
          "status": "completed",
          "harness_args": vars(harness_args),
          "summary": summary,
          "metadata": payload["metadata"],
          "method_summary": _method_summary(rows),
          "output_dir": summary["output_dir"],
          "csv_path": summary["csv_path"],
          "json_path": summary["json_path"],
      })
    except Exception as exc:  # pylint: disable=broad-except
      if args.fail_on_error:
        raise
      records.append({
          "dataset": dataset,
          "status": "unavailable",
          "harness_argv": argv,
          "error": "%s: %s" % (exc.__class__.__name__, exc),
          "traceback": traceback.format_exc(),
      })

  aggregate_csv = _write_aggregate_csv(records, output_dir)
  summary_md = _write_markdown_summary(records, output_dir)
  report = {
      "output_dir": str(output_dir),
      "aggregate_csv": str(aggregate_csv),
      "summary_md": str(summary_md),
      "records": records,
      "honesty_note": (
          "Only completed records are measured benchmark evidence. "
          "Unavailable MNIST records mean the benchmark path exists but was not run.")
  }
  report_path = output_dir / "benchmark_suite_report.json"
  report_path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
  report["report_path"] = str(report_path)
  return report


def main(argv: Optional[Sequence[str]] = None) -> None:
  args = parse_args(argv)
  report = run_suite(args)
  print(json.dumps(_json_safe({
      "output_dir": report["output_dir"],
      "aggregate_csv": report["aggregate_csv"],
      "summary_md": report["summary_md"],
      "report_path": report["report_path"],
      "statuses": {record["dataset"]: record["status"] for record in report["records"]},
  }), indent=2))


if __name__ == "__main__":
  main()
