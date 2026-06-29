# coding=utf-8
"""Run lightweight neural-network SIS experiments.

The default path is intentionally CPU-friendly:

  python -m experiments.run_nn_sis_experiments --dataset digits --model mlp

It trains a small sklearn MLP on the 8x8 digits dataset, selects correctly
classified high-confidence examples, runs baseline SIS plus extension methods,
and writes structured CSV/JSON results. Optional torchvision datasets are
handled gracefully when dependencies or downloads are unavailable.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from sufficient_input_subsets import sis
from sufficient_input_subsets.hierarchical_sis import hierarchical_sis_collection
from sufficient_input_subsets.probabilistic_sis import probabilistic_sis_collection
from sufficient_input_subsets.shap_guided_sis import (
    collection_mask,
    shap_guided_sis_collection,
)
from sufficient_input_subsets.stability_metrics import mask_f1, mask_iou


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "results" / "sis_nn_experiments"


@dataclasses.dataclass
class DatasetBundle:
  """Small in-memory dataset bundle used by the experiment harness."""

  name: str
  x_train: np.ndarray
  x_test: np.ndarray
  y_train: np.ndarray
  y_test: np.ndarray
  input_shape: Tuple[int, ...]
  n_classes: int
  preprocessing: str
  mask_baseline: str = "zero"


@dataclasses.dataclass
class ExampleSelection:
  """A correctly classified example selected for SIS evaluation."""

  test_index: int
  input_value: np.ndarray
  true_label: int
  target_class: int
  confidence: float


class CountingScoringFunction:
  """SIS-compatible black-box scoring function with evaluation counters."""

  def __init__(self, model: Any, target_class: int, input_shape: Sequence[int]):
    self.model = model
    self.target_class = int(target_class)
    self.input_shape = tuple(int(dim) for dim in input_shape)
    self.batch_calls = 0
    self.individual_model_evaluations = 0

  def reset_counts(self) -> None:
    self.batch_calls = 0
    self.individual_model_evaluations = 0

  def __call__(self, batch: np.ndarray) -> np.ndarray:
    batch = np.asarray(batch, dtype=float)
    if batch.ndim == len(self.input_shape):
      batch = np.expand_dims(batch, axis=0)
    if batch.shape[1:] != self.input_shape:
      try:
        batch = batch.reshape((-1,) + self.input_shape)
      except ValueError as exc:
        raise ValueError(
            "Expected batch shape (B, %s), got %s." %
            (self.input_shape, batch.shape)) from exc

    self.batch_calls += 1
    self.individual_model_evaluations += int(batch.shape[0])
    flat_batch = batch.reshape((batch.shape[0], -1))
    probabilities = self.model.predict_proba(flat_batch)
    return np.asarray(probabilities[:, self.target_class], dtype=float)


def _json_safe(value: Any) -> Any:
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, (np.floating, np.integer)):
    return value.item()
  if isinstance(value, Path):
    return str(value)
  if dataclasses.is_dataclass(value):
    return _json_safe(dataclasses.asdict(value))
  if isinstance(value, dict):
    return {str(k): _json_safe(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_json_safe(v) for v in value]
  return value


def load_digits_dataset(seed: int) -> DatasetBundle:
  """Loads sklearn digits as 8x8 grayscale images scaled to [0, 1]."""
  from sklearn.datasets import load_digits
  from sklearn.model_selection import train_test_split

  digits = load_digits()
  images = digits.images.astype(float) / 16.0
  labels = digits.target.astype(int)
  x_train, x_test, y_train, y_test = train_test_split(
      images,
      labels,
      test_size=0.25,
      random_state=seed,
      stratify=labels)
  return DatasetBundle(
      name="digits",
      x_train=x_train,
      x_test=x_test,
      y_train=y_train,
      y_test=y_test,
      input_shape=(8, 8),
      n_classes=10,
      preprocessing="sklearn digits pixels scaled from 0..16 to 0..1",
  )



def average_pool_downsample(images: np.ndarray, target_size: Optional[int]) -> np.ndarray:
  """Downsamples square image batches with integer-factor average pooling.

  Args:
    images: Array with shape ``(N, H, W)`` or ``(N, H, W, C)``.
    target_size: Desired square height/width. ``None`` or non-positive values
      leave the input unchanged.

  Returns:
    Downsampled images. The dtype is float because average pooling may create
    fractional values.
  """
  images = np.asarray(images, dtype=float)
  if target_size is None or int(target_size) <= 0:
    return images
  target_size = int(target_size)
  if images.ndim not in (3, 4):
    raise ValueError("average_pool_downsample expects image batches with 3 or 4 dimensions.")

  height, width = int(images.shape[1]), int(images.shape[2])
  if height == target_size and width == target_size:
    return images
  if height != width:
    raise ValueError("Only square image downsampling is supported; got %sx%s." % (height, width))
  if height % target_size != 0:
    raise ValueError(
        "Target size %d must divide the original image size %d." %
        (target_size, height))

  factor = height // target_size
  if images.ndim == 3:
    return images.reshape(
        images.shape[0], target_size, factor, target_size, factor).mean(axis=(2, 4))
  return images.reshape(
      images.shape[0], target_size, factor, target_size, factor, images.shape[-1]).mean(axis=(2, 4))


def load_openml_mnist_dataset(
    seed: int,
    train_subset: int,
    test_subset: int,
    image_size: Optional[int] = None,
) -> DatasetBundle:
  """Loads MNIST through scikit-learn/OpenML when TorchVision is unavailable."""
  from sklearn.datasets import fetch_openml

  try:
    try:
      mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    except TypeError:
      mnist = fetch_openml("mnist_784", version=1, as_frame=False)
  except Exception as exc:
    raise RuntimeError(
        "Could not load MNIST via sklearn.fetch_openml. Check network/cache "
        "availability, or install torch/torchvision for the TorchVision path.") from exc

  images = np.asarray(mnist.data, dtype=float).reshape((-1, 28, 28)) / 255.0
  labels = np.asarray(mnist.target).astype(int)
  if images.shape[0] >= 70000:
    x_train_all, y_train_all = images[:60000], labels[:60000]
    x_test_all, y_test_all = images[60000:], labels[60000:]
  else:
    split = int(0.85 * images.shape[0])
    x_train_all, y_train_all = images[:split], labels[:split]
    x_test_all, y_test_all = images[split:], labels[split:]

  rng = np.random.default_rng(seed)
  train_indices = rng.choice(
      x_train_all.shape[0], size=min(train_subset, x_train_all.shape[0]), replace=False)
  test_indices = rng.choice(
      x_test_all.shape[0], size=min(test_subset, x_test_all.shape[0]), replace=False)
  x_train = average_pool_downsample(x_train_all[train_indices], image_size)
  x_test = average_pool_downsample(x_test_all[test_indices], image_size)
  preprocessing = "OpenML mnist_784 pixels scaled from 0..255 to 0..1; small random subset"
  if image_size is not None and int(image_size) > 0:
    preprocessing += "; average-pooled to %dx%d" % (int(image_size), int(image_size))
  return DatasetBundle(
      name="mnist",
      x_train=x_train,
      x_test=x_test,
      y_train=y_train_all[train_indices].astype(int),
      y_test=y_test_all[test_indices].astype(int),
      input_shape=tuple(x_train.shape[1:]),
      n_classes=10,
      preprocessing=preprocessing,
  )

def load_torchvision_dataset(
    dataset_name: str,
    seed: int,
    train_subset: int,
    test_subset: int,
    image_size: Optional[int] = None,
) -> DatasetBundle:
  """Loads optional torchvision datasets into small NumPy arrays.

  This path is intentionally conservative. It raises a clear RuntimeError if
  torch/torchvision is unavailable or dataset download fails.
  """
  try:
    import torch  # type: ignore
    from torchvision import datasets, transforms  # type: ignore
  except Exception as exc:
    if dataset_name == "mnist":
      try:
        return load_openml_mnist_dataset(
            seed=seed,
            train_subset=train_subset,
            test_subset=test_subset,
            image_size=image_size)
      except RuntimeError as openml_exc:
        raise RuntimeError(
            "MNIST could not be loaded through TorchVision or OpenML. "
            "TorchVision error: %s. OpenML error: %s" %
            (exc, openml_exc)) from openml_exc
    raise RuntimeError(
        "Torch/TorchVision is required for dataset '%s'. Use '--dataset digits' "
        "for the dependency-light path." % dataset_name) from exc

  dataset_map = {
      "mnist": datasets.MNIST,
      "fashion_mnist": datasets.FashionMNIST,
      "cifar10": datasets.CIFAR10,
  }
  if dataset_name not in dataset_map:
    raise ValueError("Unsupported torchvision dataset: %s" % dataset_name)

  transform = transforms.ToTensor()
  root = ROOT / "data" / "torchvision"
  try:
    train_ds = dataset_map[dataset_name](root=str(root), train=True, download=True, transform=transform)
    test_ds = dataset_map[dataset_name](root=str(root), train=False, download=True, transform=transform)
  except Exception as exc:
    if dataset_name == "mnist":
      try:
        return load_openml_mnist_dataset(
            seed=seed,
            train_subset=train_subset,
            test_subset=test_subset,
            image_size=image_size)
      except RuntimeError as openml_exc:
        raise RuntimeError(
            "MNIST could not be loaded through TorchVision or OpenML. "
            "TorchVision error: %s. OpenML error: %s" %
            (exc, openml_exc)) from openml_exc
    raise RuntimeError(
        "Could not download/load '%s'. Check network availability or use digits." %
        dataset_name) from exc

  rng = np.random.default_rng(seed)
  train_indices = rng.choice(len(train_ds), size=min(train_subset, len(train_ds)), replace=False)
  test_indices = rng.choice(len(test_ds), size=min(test_subset, len(test_ds)), replace=False)

  def collect(ds: Any, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for idx in indices:
      tensor, label = ds[int(idx)]
      array = tensor.numpy()
      if array.shape[0] == 1:
        array = array[0]
      else:
        array = np.transpose(array, (1, 2, 0))
      xs.append(array.astype(float))
      ys.append(int(label))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=int)

  x_train, y_train = collect(train_ds, train_indices)
  x_test, y_test = collect(test_ds, test_indices)
  x_train = average_pool_downsample(x_train, image_size)
  x_test = average_pool_downsample(x_test, image_size)
  preprocessing = "torchvision ToTensor values in 0..1; small random subset"
  if image_size is not None and int(image_size) > 0:
    preprocessing += "; average-pooled to %dx%d" % (int(image_size), int(image_size))
  return DatasetBundle(
      name=dataset_name,
      x_train=x_train,
      x_test=x_test,
      y_train=y_train,
      y_test=y_test,
      input_shape=tuple(x_train.shape[1:]),
      n_classes=10,
      preprocessing=preprocessing,
  )


def load_dataset(args: argparse.Namespace) -> DatasetBundle:
  if args.dataset == "digits":
    return load_digits_dataset(seed=args.seed)
  return load_torchvision_dataset(
      args.dataset,
      seed=args.seed,
      train_subset=args.train_subset,
      test_subset=args.test_subset,
      image_size=args.image_size)


def train_mlp_model(dataset: DatasetBundle, seed: int, max_iter: int):
  """Trains a small sklearn MLPClassifier."""
  from sklearn.neural_network import MLPClassifier
  from sklearn.pipeline import make_pipeline
  from sklearn.preprocessing import StandardScaler

  x_train_flat = dataset.x_train.reshape((dataset.x_train.shape[0], -1))
  model = make_pipeline(
      StandardScaler(),
      MLPClassifier(
          hidden_layer_sizes=(64,),
          activation="relu",
          solver="adam",
          alpha=1e-4,
          batch_size=64,
          learning_rate_init=1e-3,
          max_iter=int(max_iter),
          random_state=seed,
          early_stopping=True,
          n_iter_no_change=10))
  model.fit(x_train_flat, dataset.y_train)
  return model


def train_model(args: argparse.Namespace, dataset: DatasetBundle):
  """Trains the requested lightweight model."""
  if args.model == "mlp":
    return train_mlp_model(dataset, seed=args.seed, max_iter=args.max_iter)
  if args.model == "small_cnn":
    raise RuntimeError(
        "small_cnn is reserved for a future PyTorch path. The current measured "
        "CPU-friendly harness supports '--model mlp'.")
  raise ValueError("Unsupported model: %s" % args.model)


def model_accuracy(model: Any, x: np.ndarray, y: np.ndarray) -> float:
  predictions = model.predict(x.reshape((x.shape[0], -1)))
  return float(np.mean(predictions == y))


def select_examples(
    model: Any,
    dataset: DatasetBundle,
    max_examples: int,
    min_confidence: float,
) -> List[ExampleSelection]:
  """Selects correctly classified high-confidence test examples."""
  x_flat = dataset.x_test.reshape((dataset.x_test.shape[0], -1))
  probabilities = model.predict_proba(x_flat)
  predictions = np.argmax(probabilities, axis=1)
  confidences = np.max(probabilities, axis=1)
  candidates = [
      idx for idx in range(dataset.x_test.shape[0])
      if predictions[idx] == dataset.y_test[idx] and confidences[idx] >= min_confidence
  ]
  if not candidates:
    candidates = [
        idx for idx in range(dataset.x_test.shape[0])
        if predictions[idx] == dataset.y_test[idx]
    ]
  ordered = sorted(candidates, key=lambda idx: float(confidences[idx]), reverse=True)
  selected = []
  for idx in ordered[:max(1, int(max_examples))]:
    selected.append(ExampleSelection(
        test_index=int(idx),
        input_value=np.asarray(dataset.x_test[idx], dtype=float),
        true_label=int(dataset.y_test[idx]),
        target_class=int(predictions[idx]),
        confidence=float(confidences[idx]),
    ))
  return selected


def choose_threshold(confidence: float, args: argparse.Namespace) -> float:
  if args.threshold_mode == "fixed":
    return float(args.threshold)
  if args.threshold_mode == "relative":
    return float(max(args.min_threshold, min(float(args.threshold), args.relative_fraction * confidence)))
  raise ValueError("threshold_mode must be fixed or relative.")


def _collection_union_mask(collection: Sequence[sis.SISResult], input_shape: Sequence[int]) -> np.ndarray:
  if not collection:
    return np.zeros(tuple(input_shape), dtype=bool)
  return collection_mask(collection)


def _score_mask(f_batch: CountingScoringFunction, x: np.ndarray, baseline: np.ndarray, mask: np.ndarray) -> float:
  if mask.size == 0:
    return float("nan")
  masked_input = sis.produce_masked_inputs(x, baseline, np.asarray([mask]))
  return float(np.asarray(f_batch(masked_input), dtype=float).reshape(-1)[0])


def _run_original_sis(
    f_batch: CountingScoringFunction,
    threshold: float,
    x: np.ndarray,
    baseline: np.ndarray,
) -> Tuple[List[sis.SISResult], Dict[str, Any]]:
  start = time.perf_counter()
  collection = sis.sis_collection(f_batch, threshold, x, baseline)
  runtime = time.perf_counter() - start
  mask = _collection_union_mask(collection, x.shape)
  return collection, {
      "runtime": runtime,
      "model_call_count": f_batch.batch_calls,
      "individual_model_evaluations": f_batch.individual_model_evaluations,
      "subset_size": int(np.sum(mask)),
      "sufficiency_score": _score_mask(f_batch, x, baseline, mask),
      "threshold_met": bool(collection),
      "n_sis": len(collection),
  }


def run_methods_for_example(
    model: Any,
    dataset: DatasetBundle,
    example: ExampleSelection,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
  """Runs SIS variants for one selected example."""
  x = np.asarray(example.input_value, dtype=float)
  baseline = np.zeros_like(x)
  threshold = choose_threshold(example.confidence, args)
  rows: List[Dict[str, Any]] = []
  masks: Dict[str, np.ndarray] = {}

  f_original = CountingScoringFunction(model, example.target_class, dataset.input_shape)
  original_collection, original_diag = _run_original_sis(f_original, threshold, x, baseline)
  original_mask = _collection_union_mask(original_collection, x.shape)
  masks["original_sis"] = original_mask
  baseline_evals = max(1, int(original_diag["individual_model_evaluations"]))
  baseline_runtime = max(1e-12, float(original_diag["runtime"]))
  rows.append(_row_from_diag(
      dataset, example, threshold, "original_sis", original_diag,
      baseline_evals=baseline_evals, baseline_runtime=baseline_runtime))

  f_shap = CountingScoringFunction(model, example.target_class, dataset.input_shape)
  collection, shap_diag = shap_guided_sis_collection(
      f_shap,
      threshold,
      x,
      baseline,
      batch_size=args.batch_size,
      max_candidates=args.max_candidates,
      baseline_runtime=baseline_runtime,
      baseline_model_evaluations=baseline_evals,
      return_diagnostics=True,
      random_state=args.seed)
  shap_mask = _collection_union_mask(collection, x.shape)
  masks["shap_guided_sis"] = shap_mask
  shap_diag = dict(shap_diag)
  shap_diag["sufficiency_score"] = _score_mask(f_shap, x, baseline, shap_mask)
  shap_diag["threshold_met"] = bool(shap_diag["sufficiency_score"] >= threshold)
  rows.append(_row_from_diag(
      dataset, example, threshold, "shap_guided_sis", shap_diag,
      baseline_evals=baseline_evals, baseline_runtime=baseline_runtime))

  if not args.skip_probabilistic:
    f_prob = CountingScoringFunction(model, example.target_class, dataset.input_shape)
    start = time.perf_counter()
    prob = probabilistic_sis_collection(
        f_prob,
        threshold,
        x,
        baseline,
        n_samples=args.probabilistic_samples,
        noise_scale=args.probabilistic_noise,
        random_state=args.seed)
    runtime = time.perf_counter() - start
    prob_mask = np.asarray(prob.inclusion_probabilities >= args.probabilistic_mask_threshold, dtype=bool)
    masks["probabilistic_sis"] = prob_mask
    prob_diag = {
        "runtime": runtime,
        "model_call_count": f_prob.batch_calls,
        "individual_model_evaluations": f_prob.individual_model_evaluations,
        "subset_size": int(np.sum(prob_mask)),
        "sufficiency_score": _score_mask(f_prob, x, baseline, prob_mask),
        "threshold_met": False,
        "n_sis": len(prob.sampled_results),
        "mean_subset_size": prob.mean_subset_size,
        "subset_size_variance": prob.variance_subset_size,
    }
    prob_diag["threshold_met"] = bool(prob_diag["sufficiency_score"] >= threshold)
    rows.append(_row_from_diag(
        dataset, example, threshold, "probabilistic_sis", prob_diag,
        baseline_evals=baseline_evals, baseline_runtime=baseline_runtime))

  if not args.skip_hierarchical:
    f_hier = CountingScoringFunction(model, example.target_class, dataset.input_shape)
    start = time.perf_counter()
    levels = tuple(int(level) for level in args.hierarchy_levels.split(",") if level.strip())
    hier = hierarchical_sis_collection(
        f_hier,
        threshold,
        x,
        baseline,
        levels=levels,
        mode=args.hierarchy_mode)
    runtime = time.perf_counter() - start
    if hier.final_masks:
      hier_mask = np.zeros_like(hier.final_masks[0], dtype=bool)
      for mask in hier.final_masks:
        hier_mask = np.logical_or(hier_mask, np.asarray(mask, dtype=bool))
    elif hier.level_masks:
      hier_mask = np.asarray(hier.level_masks[-1], dtype=bool)
    else:
      hier_mask = np.zeros_like(x, dtype=bool)
    masks["hierarchical_sis"] = hier_mask
    hier_diag = {
        "runtime": runtime,
        "model_call_count": f_hier.batch_calls,
        "individual_model_evaluations": f_hier.individual_model_evaluations,
        "subset_size": int(np.sum(hier_mask)),
        "sufficiency_score": _score_mask(f_hier, x, baseline, hier_mask),
        "threshold_met": False,
        "n_sis": len(hier.final_masks),
    }
    hier_diag["threshold_met"] = bool(hier_diag["sufficiency_score"] >= threshold)
    rows.append(_row_from_diag(
        dataset, example, threshold, "hierarchical_sis", hier_diag,
        baseline_evals=baseline_evals, baseline_runtime=baseline_runtime))

  if args.stability_perturbations > 0:
    stability_rows = compute_stability_rows(
        model, dataset, example, threshold, baseline, masks, args)
    for row in rows:
      stability = stability_rows.get(row["method"], {})
      row.update(stability)

  return rows, masks


def _row_from_diag(
    dataset: DatasetBundle,
    example: ExampleSelection,
    threshold: float,
    method: str,
    diag: Dict[str, Any],
    baseline_evals: int,
    baseline_runtime: float,
) -> Dict[str, Any]:
  evals = int(diag.get("individual_model_evaluations", 0) or 0)
  runtime = float(diag.get("runtime", 0.0) or 0.0)
  return {
      "dataset": dataset.name,
      "model_type": "mlp",
      "example_index": example.test_index,
      "target_class": example.target_class,
      "true_label": example.true_label,
      "original_confidence": example.confidence,
      "threshold": float(threshold),
      "method": method,
      "threshold_satisfied": bool(diag.get("threshold_met", False)),
      "final_confidence": float(diag.get("sufficiency_score", float("nan"))),
      "subset_size": int(diag.get("subset_size", diag.get("selected_feature_count", 0)) or 0),
      "individual_model_evaluations": evals,
      "batched_function_calls": int(diag.get("model_call_count", 0) or 0),
      "wall_clock_runtime_sec": runtime,
      "evaluation_reduction_vs_baseline_pct": (
          100.0 * (baseline_evals - evals) / baseline_evals if baseline_evals else 0.0),
      "runtime_reduction_vs_baseline_pct": (
          100.0 * (baseline_runtime - runtime) / baseline_runtime if baseline_runtime else 0.0),
      "notes": "",
  }


def compute_stability_rows(
    model: Any,
    dataset: DatasetBundle,
    example: ExampleSelection,
    threshold: float,
    baseline: np.ndarray,
    clean_masks: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> Dict[str, Dict[str, Any]]:
  """Runs a lightweight perturbation stability probe for existing masks."""
  rng = np.random.default_rng(args.seed + 1000 + example.test_index)
  x = np.asarray(example.input_value, dtype=float)
  stability: Dict[str, Dict[str, Any]] = {
      method: {
          "stability_mask_iou": float("nan"),
          "stability_mask_f1": float("nan"),
          "stability_subset_size_difference": float("nan"),
          "stability_confidence_retention": float("nan"),
          "stability_explanation_drift": float("nan"),
      }
      for method in clean_masks
  }

  aggregate: Dict[str, List[Dict[str, float]]] = {method: [] for method in clean_masks}
  for _ in range(args.stability_perturbations):
    perturbed = np.clip(
        x + rng.normal(0.0, args.stability_noise, size=x.shape), 0.0, 1.0)
    for method, clean_mask in clean_masks.items():
      pert_mask = run_single_method_mask(
          method, model, dataset, example.target_class, perturbed, baseline,
          threshold, args)
      f_pert_score = CountingScoringFunction(model, example.target_class, dataset.input_shape)
      pert_conf = _score_mask(f_pert_score, perturbed, baseline, pert_mask)
      clean_conf = _score_mask(
          CountingScoringFunction(model, example.target_class, dataset.input_shape),
          x,
          baseline,
          clean_mask)
      iou = mask_iou(clean_mask, pert_mask)
      f1 = mask_f1(clean_mask, pert_mask)
      aggregate[method].append({
          "iou": iou,
          "f1": f1,
          "subset_diff": abs(float(np.sum(clean_mask)) - float(np.sum(pert_mask))),
          "confidence_retention": pert_conf / clean_conf if clean_conf else float("nan"),
          "drift": 1.0 - iou,
      })

  for method, entries in aggregate.items():
    if not entries:
      continue
    stability[method] = {
        "stability_mask_iou": float(np.mean([entry["iou"] for entry in entries])),
        "stability_mask_f1": float(np.mean([entry["f1"] for entry in entries])),
        "stability_subset_size_difference": float(np.mean([entry["subset_diff"] for entry in entries])),
        "stability_confidence_retention": float(np.mean([entry["confidence_retention"] for entry in entries])),
        "stability_explanation_drift": float(np.mean([entry["drift"] for entry in entries])),
      }
  return stability




def run_single_method_mask(
    method: str,
    model: Any,
    dataset: DatasetBundle,
    target_class: int,
    x: np.ndarray,
    baseline: np.ndarray,
    threshold: float,
    args: argparse.Namespace,
) -> np.ndarray:
  """Recomputes one method on a perturbed input and returns its explanation mask."""
  f_batch = CountingScoringFunction(model, target_class, dataset.input_shape)
  if method == "original_sis":
    collection = sis.sis_collection(f_batch, threshold, x, baseline)
    return _collection_union_mask(collection, x.shape)
  if method == "shap_guided_sis":
    collection, _ = shap_guided_sis_collection(
        f_batch,
        threshold,
        x,
        baseline,
        batch_size=args.batch_size,
        max_candidates=args.max_candidates,
        return_diagnostics=True,
        random_state=args.seed)
    return _collection_union_mask(collection, x.shape)
  if method == "probabilistic_sis":
    prob = probabilistic_sis_collection(
        f_batch,
        threshold,
        x,
        baseline,
        n_samples=args.probabilistic_samples,
        noise_scale=args.probabilistic_noise,
        random_state=args.seed)
    return np.asarray(
        prob.inclusion_probabilities >= args.probabilistic_mask_threshold,
        dtype=bool)
  if method == "hierarchical_sis":
    levels = tuple(int(level) for level in args.hierarchy_levels.split(",") if level.strip())
    hier = hierarchical_sis_collection(
        f_batch,
        threshold,
        x,
        baseline,
        levels=levels,
        mode=args.hierarchy_mode)
    if hier.final_masks:
      union = np.zeros_like(hier.final_masks[0], dtype=bool)
      for mask in hier.final_masks:
        union = np.logical_or(union, np.asarray(mask, dtype=bool))
      return union
    if hier.level_masks:
      return np.asarray(hier.level_masks[-1], dtype=bool)
    return np.zeros_like(x, dtype=bool)
  raise ValueError("Unknown method for stability recomputation: %s" % method)
def write_outputs(
    rows: Sequence[Dict[str, Any]],
    metadata: Dict[str, Any],
    output_dir: Path,
) -> Tuple[Path, Path]:
  """Writes CSV and JSON experiment results."""
  output_dir.mkdir(parents=True, exist_ok=True)
  csv_path = output_dir / "results.csv"
  json_path = output_dir / "results.json"
  if rows:
    fieldnames = list(rows[0].keys())
    for row in rows:
      for key in row:
        if key not in fieldnames:
          fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
      writer = csv.DictWriter(f, fieldnames=fieldnames)
      writer.writeheader()
      for row in rows:
        writer.writerow(row)
  else:
    csv_path.write_text("", encoding="utf-8")

  payload = {
      "metadata": metadata,
      "results": list(rows),
  }
  json_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
  return csv_path, json_path


def make_output_dir(
    base_output_dir: Optional[str],
    dataset: str = "digits",
    model: str = "mlp",
) -> Path:
  stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  if base_output_dir:
    path = Path(base_output_dir)
    if not path.is_absolute():
      path = ROOT / path
    if (path / "results.csv").exists() or (path / "results.json").exists():
      return path.parent / ("%s_%s" % (path.name, stamp))
    return path
  return DEFAULT_RESULTS_DIR / ("%s_%s_%s" % (dataset, model, stamp))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--dataset", choices=["digits", "mnist", "fashion_mnist", "cifar10"], default="digits")
  parser.add_argument("--model", choices=["mlp", "small_cnn"], default="mlp")
  parser.add_argument("--max-examples", type=int, default=1)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--threshold-mode", choices=["fixed", "relative"], default="relative")
  parser.add_argument("--threshold", type=float, default=0.8)
  parser.add_argument("--relative-fraction", type=float, default=0.85)
  parser.add_argument("--min-threshold", type=float, default=0.5)
  parser.add_argument("--min-confidence", type=float, default=0.70)
  parser.add_argument("--output-dir", type=str, default=None)
  parser.add_argument("--max-iter", type=int, default=200)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--max-candidates", type=int, default=None)
  parser.add_argument("--probabilistic-samples", type=int, default=3)
  parser.add_argument("--probabilistic-noise", type=float, default=0.05)
  parser.add_argument("--probabilistic-mask-threshold", type=float, default=0.5)
  parser.add_argument("--hierarchy-levels", type=str, default="2,4,8")
  parser.add_argument("--hierarchy-mode", choices=["grid", "pixel"], default="pixel")
  parser.add_argument("--stability-perturbations", type=int, default=2)
  parser.add_argument("--stability-noise", type=float, default=0.02)
  parser.add_argument("--skip-probabilistic", action="store_true")
  parser.add_argument("--skip-hierarchical", action="store_true")
  parser.add_argument("--train-subset", type=int, default=2000)
  parser.add_argument("--test-subset", type=int, default=500)
  parser.add_argument(
      "--image-size",
      type=int,
      default=None,
      help=(
          "Optional square downsample size for torchvision image datasets. "
          "For example, use 14 for a CPU-friendly MNIST SIS benchmark."))
  return parser.parse_args(argv)


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
  np.random.seed(args.seed)
  output_dir = make_output_dir(args.output_dir, dataset=args.dataset, model=args.model)
  start = time.perf_counter()
  dataset = load_dataset(args)
  model = train_model(args, dataset)
  train_accuracy = model_accuracy(model, dataset.x_train, dataset.y_train)
  test_accuracy = model_accuracy(model, dataset.x_test, dataset.y_test)
  examples = select_examples(
      model, dataset, max_examples=args.max_examples, min_confidence=args.min_confidence)
  if not examples:
    raise RuntimeError("No correctly classified examples found for SIS evaluation.")

  rows: List[Dict[str, Any]] = []
  for example in examples:
    example_rows, _ = run_methods_for_example(model, dataset, example, args)
    rows.extend(example_rows)

  metadata = {
      "dataset": {
          "name": dataset.name,
          "input_shape": dataset.input_shape,
          "n_classes": dataset.n_classes,
          "n_train": int(dataset.x_train.shape[0]),
          "n_test": int(dataset.x_test.shape[0]),
          "preprocessing": dataset.preprocessing,
          "mask_baseline": dataset.mask_baseline,
      },
      "model_type": args.model,
      "model_library": "sklearn.neural_network.MLPClassifier" if args.model == "mlp" else args.model,
      "train_accuracy": train_accuracy,
      "test_accuracy": test_accuracy,
      "selected_examples": [dataclasses.asdict(example) for example in examples],
      "args": vars(args),
      "runtime_sec": time.perf_counter() - start,
      "honesty_note": (
          "These are measured local results for the configured run only. "
          "Do not claim broader MNIST/CIFAR/CNN results unless those runs are executed.")
  }
  csv_path, json_path = write_outputs(rows, metadata, output_dir)
  return {
      "output_dir": str(output_dir),
      "csv_path": str(csv_path),
      "json_path": str(json_path),
      "n_rows": len(rows),
      "train_accuracy": train_accuracy,
      "test_accuracy": test_accuracy,
      "selected_examples": len(examples),
  }


def main(argv: Optional[Sequence[str]] = None) -> None:
  args = parse_args(argv)
  summary = run_experiment(args)
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()


