# SIS Research Extensions

This project extends Google Research's NumPy implementation of Sufficient Input
Subsets (SIS) in `sis.py`. The original API is preserved: `sis_collection`
takes a batched black-box function `f_batch`, a confidence `threshold`, an
`initial_input`, a `fully_masked_input`, and an optional broadcastable
`initial_mask`. It returns a list of `SISResult` objects with:

- `sis`: indices of unmasked features in the sufficient subset
- `ordering_over_entire_backselect`: feature order considered during search
- `values_over_entire_backselect`: model values observed during search
- `mask`: boolean mask where `True` means the feature is present

## What This Adds

- `shap_guided_sis.py`: SHAP-inspired feature ranking, batched perturbation
  scoring, masked-input memoization, and SIS-compatible results plus diagnostics.
- `probabilistic_sis.py`: repeated stochastic SIS sampling and per-feature
  inclusion probabilities.
- `hierarchical_sis.py`: coarse-to-fine explanations over grids, pixels, and
  optional superpixels when `scikit-image` is available.
- `stability_metrics.py`: IoU, F1, subset-size variance, perturbation drift,
  confidence retention, and adversarial sensitivity summaries.
- `benchmark_sis.py`: measured comparison of original SIS, SHAP-guided SIS,
  Probabilistic SIS, and Hierarchical SIS.
- `vision_demo.py`: small runnable image demo using `sklearn` digits when
  available, with a synthetic fallback.

## SHAP-Guided Acceleration

The default importance estimator masks one feature or feature group at a time
and measures the confidence drop:

```python
importance = f(x_current) - f(x_with_group_masked)
```

Features are scored in batches through `f_batch`. Repeated masked inputs are
memoized, so construction and pruning reuse cached evaluations when possible.
The wrapper then builds a sufficient subset by adding high-importance features
until the threshold is met, and prunes removable features while preserving
sufficiency.

Use it as:

```python
from sufficient_input_subsets.shap_guided_sis import shap_guided_sis_collection

collection, diagnostics = shap_guided_sis_collection(
    f_batch,
    threshold,
    initial_input,
    fully_masked_input,
    batch_size=64,
    return_diagnostics=True,
)
```

Set `return_diagnostics=False` to receive only the SIS collection, matching the
shape of the original `sis_collection` return value.

## Probabilistic SIS

Probabilistic SIS samples multiple plausible explanations by adding controlled
noise to the feature ranking. Across samples, it estimates how often each
feature is included.

Outputs include:

- sampled SIS collections
- inclusion probability map
- mean and variance of subset size
- confidence and stability summaries

```python
from sufficient_input_subsets.probabilistic_sis import probabilistic_sis_collection

result = probabilistic_sis_collection(
    f_batch,
    threshold,
    initial_input,
    fully_masked_input,
    n_samples=30,
    noise_scale=0.05,
    random_state=0,
)
```

## Hierarchical SIS

Hierarchical SIS starts from coarse regions and refines important areas into
smaller regions or pixels. Grid mode works without extra dependencies. Pixel
mode uses grid levels until the last level, then refines to individual pixels.

```python
from sufficient_input_subsets.hierarchical_sis import hierarchical_sis_collection

result = hierarchical_sis_collection(
    f_batch,
    threshold,
    initial_input,
    fully_masked_input,
    levels=(4, 8, 16),
    mode="grid",
)
```

The result stores `tree`, `level_masks`, `final_masks`, and `diagnostics`.

## Stability Metrics

The stability module quantifies explanation robustness:

- `mask_iou(mask_a, mask_b)`: pairwise intersection-over-union
- `mask_f1(mask_a, mask_b)`: pairwise F1 overlap
- `explanation_stability(masks)`: pairwise stability and subset-size variance
- `perturbation_stability(...)`: drift and confidence retention under input noise
- `adversarial_sensitivity_score(...)`: combined drift and confidence-drop score

High explanation drift under small input perturbations is a useful baseline
signal for robustness analysis and adversarial vulnerability investigation.

## Running Tests

From the repository root:

```bash
python -m unittest discover -s sufficient_input_subsets/tests -p test*.py
python -m sufficient_input_subsets.sis_test
```

The original Google test imports `absl.testing`. This workspace includes a tiny
local compatibility shim so tests can run without installing `absl-py`. In a
normal environment, installing `absl-py` from `requirements.txt` is also fine.

## Running The Demo

```bash
python -m sufficient_input_subsets.vision_demo
```

The demo saves:

- `results/vision_demo_original.png`
- `results/vision_demo_baseline_sis_mask.png`
- `results/vision_demo_shap_guided_sis_mask.png`
- `results/vision_demo_probabilistic_heatmap.png`
- `results/vision_demo_hierarchical_regions.png`
- `results/vision_demo_perturbation_stability.png`
- `results/vision_demo_summary.json`

## Running Benchmarks

```bash
python -m sufficient_input_subsets.benchmark_sis
```

This writes:

- `results/benchmark_report.json`
- `results/benchmark_summary.md`

Measured on the included synthetic image benchmark:

- Original SIS: 3974 individual model evaluations
- SHAP-guided SIS: 142 individual model evaluations
- Measured SHAP-guided reduction: 96.43%

This exceeds the resume target of "approximately 20%" on the synthetic
benchmark. Do not claim this number universally. On a new model or dataset, use
the benchmark script and report the actual measured reduction. If the reduction
falls below target, tune `max_candidates`, use region-level feature groups, and
reuse caches across repeated perturbation runs.

## Interview Explanation

60-second version: I extended Google Research's SIS implementation into a small
research-engineering framework for image explanations. The original SIS finds a
minimal subset of input features that keeps a model prediction above a
confidence threshold. I preserved that API, then added a SHAP-inspired
importance pass that ranks features before SIS search, probabilistic sampling
to estimate uncertainty in explanations, hierarchical grid-to-pixel refinement
for multi-scale masks, and stability metrics that measure how much explanations
drift under perturbations.

Problem solved: vanilla SIS is faithful but expensive because backward
selection repeatedly evaluates many masked variants. The acceleration reduces
unnecessary evaluations by prioritizing features likely to matter.

Technically difficult: keeping the outputs compatible with `SISResult` while
adding grouped features, caching, stochastic sampling, and hierarchy-level
diagnostics without rewriting the original algorithm.

How the 20% reduction is measured: `benchmark_sis.py` wraps `f_batch` with a
counter and compares individual model evaluations and wall-clock runtime
against original `sis_collection`. The generated benchmark report records the
actual speedup; the current synthetic benchmark measured 96.43%.

Robustness link: stable explanations should remain similar under small,
non-semantic perturbations. Large mask drift or confidence loss can flag inputs
where the model's decision rationale is brittle, which is a useful starting
point for adversarial vulnerability analysis.

Limitations and future work: the SHAP fallback is perturbation-based rather
than exact SHAP, grid hierarchy is simpler than semantic segmentation, and the
benchmark is lightweight. Stronger validation should run on real vision models,
larger datasets, and adversarial attacks such as FGSM or PGD.
