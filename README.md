# SIS Research Work

This repository extends Google Research's Sufficient Input Subsets (SIS)
implementation with research-engineering tools for model-agnostic explainability.
SIS can explain neural networks through a black-box scoring function `f_batch`,
so the code keeps the original NumPy SIS implementation and adds experiment
wrappers around it rather than rewriting the core algorithm.

## What Is Included

- Original SIS-style sufficient subset search in `sufficient_input_subsets/sis.py`
- SHAP-inspired / perturbation-guided SIS acceleration
- Probabilistic SIS for inclusion-probability explanations
- Hierarchical SIS for coarse-to-fine image masks
- Stability metrics for perturbation robustness
- Synthetic 8x8 image-style benchmark
- sklearn digits demo
- CPU-friendly neural-network experiment harness for sklearn digits and MNIST + MLP

## Neural-Network Experiment Setup

The main neural-network entry point is:

```bash
python -m experiments.run_nn_sis_experiments --dataset digits --model mlp --max-examples 1 --seed 0 --output-dir results/sis_nn_experiments
```

This command trains a small sklearn `MLPClassifier` on the sklearn digits
dataset and runs:

- original SIS
- SHAP-guided SIS
- Probabilistic SIS
- Hierarchical SIS
- lightweight perturbation stability probes

Outputs:

- `results/sis_nn_experiments/results.csv`
- `results/sis_nn_experiments/results.json`

If the requested output directory already contains result files, the harness
creates a timestamped sibling directory instead of silently overwriting it.


## Neural-Network Benchmark Suite

To run a small benchmark suite that includes both sklearn digits and MNIST:

```bash
python -m experiments.run_nn_benchmark_suite --datasets digits,mnist --max-examples 1 --max-iter 80 --mnist-train-subset 2000 --mnist-test-subset 500 --mnist-image-size 14 --stability-perturbations 0 --output-dir results/sis_nn_benchmarks
```

The suite writes an aggregate CSV, JSON report, and Markdown summary under a
timestamped directory such as `results/sis_nn_benchmarks/benchmark_suite_*`.
MNIST is loaded through TorchVision when available, with a scikit-learn OpenML
fallback. If neither path works, the suite records MNIST as `unavailable`
instead of fabricating results.

## Dataset Setup

### `digits` default

- Source: `sklearn.datasets.load_digits`
- Shape: 8x8 grayscale images
- Classes: 10
- Preprocessing: pixels scaled from `0..16` to `[0, 1]`
- Split: stratified train/test split with `test_size=0.25`
- Model: sklearn `MLPClassifier`
- Mask baseline: zero image
- Example selection: correctly classified high-confidence test examples

### MNIST and optional torchvision datasets

The harness accepts `--dataset mnist`, `--dataset fashion_mnist`, and
`--dataset cifar10`.

- MNIST can be loaded through TorchVision or, when PyTorch is unavailable,
  through `sklearn.datasets.fetch_openml("mnist_784")`.
- The benchmark suite defaults to `--mnist-image-size 14`, which average-pools
  28x28 MNIST images to 14x14 so baseline SIS remains CPU-friendly.
- Fashion-MNIST and CIFAR-10 still require Torch/TorchVision and should be
  claimed only after their scripts complete and save result files.
- All dataset loaders use a zero image as the masking baseline.

## Methodology

For each selected example, the harness exposes the trained model through an
SIS-compatible black-box scoring function:

```python
f_batch(masked_images) -> confidence_for_target_class
```

It records:

- dataset
- model type
- example index
- target class
- original confidence
- threshold
- method name
- threshold satisfaction
- final confidence
- subset size
- individual model evaluations
- batched function calls
- wall-clock runtime
- evaluation reduction vs baseline
- perturbation stability metrics

Threshold policy defaults to `relative`: `min(threshold, relative_fraction *
original_confidence)`, with a lower bound controlled by `--min-threshold`.
The default command uses threshold `0.8`, relative fraction `0.85`, and minimum
threshold `0.5`.

## Current Measured Results

The latest checked run is stored in `results/sis_nn_experiments/results.json`.
It used sklearn digits, an MLP, one selected high-confidence example, and seed 0.

Measured model performance:

- Train accuracy: `0.9911`
- Test accuracy: `0.9733`
- Selected example: test index `187`, target class `4`
- Original confidence: `0.9999985`
- SIS threshold: `0.8`

Measured SIS results:

| Method | Evaluations | f_batch calls | Runtime (s) | Subset size | Final confidence | Eval reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original SIS | 7945 | 255 | 0.0432 | 4 | 0.999998 | 0.00% |
| SHAP-guided SIS | 199 | 13 | 0.0035 | 3 | 0.998905 | 97.50% |
| Probabilistic SIS | 600 | 42 | 0.0114 | 3 | 0.998905 | 92.45% |
| Hierarchical SIS | 40 | 23 | 0.0040 | 2 | 0.963973 | 99.50% |

These are local measured results for this specific lightweight run.

### Digits + MNIST benchmark suite

A measured suite run is stored at
`results/sis_nn_benchmarks/benchmark_suite_20260628_233409/`. It used one
selected example per dataset, sklearn `MLPClassifier`, threshold `0.8`, zero
masking, no stability perturbations, and MNIST downsampled to 14x14.

| Dataset | Test accuracy | Method | Evaluations | Runtime (s) | Subset size | Final confidence | Eval reduction |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| digits | 0.9733 | Original SIS | 7945 | 0.0506 | 4 | 0.999998 | 0.00% |
| digits | 0.9733 | SHAP-guided SIS | 199 | 0.0036 | 3 | 0.998905 | 97.50% |
| digits | 0.9733 | Probabilistic SIS | 600 | 0.0113 | 3 | 0.998905 | 92.45% |
| digits | 0.9733 | Hierarchical SIS | 40 | 0.0043 | 2 | 0.963973 | 99.50% |
| MNIST 14x14 | 0.8940 | Original SIS | 56370 | 0.2341 | 10 | 0.998483 | 0.00% |
| MNIST 14x14 | 0.8940 | SHAP-guided SIS | 405 | 0.0098 | 6 | 0.994795 | 99.28% |
| MNIST 14x14 | 0.8940 | Probabilistic SIS | 1218 | 0.0280 | 6 | 0.994795 | 97.84% |
| MNIST 14x14 | 0.8940 | Hierarchical SIS | 121 | 0.0093 | 2 | 0.821547 | 99.79% |

These are measured small-benchmark results, not broad CNN/CIFAR/adversarial
claims.

## Running Tests

```bash
python -m unittest discover -s sufficient_input_subsets/tests -p test*.py
python -m sufficient_input_subsets.sis_test
```

Latest observed status:

- Extension, harness, and benchmark-suite tests: 20 tests passed
- Original SIS tests: 18 tests passed

## Other Runnable Commands

Synthetic benchmark:

```bash
python -m sufficient_input_subsets.benchmark_sis
```

Vision demo:

```bash
python -m sufficient_input_subsets.vision_demo --n_probabilistic_samples 3
```

## Honest Scope

Implemented and measured:

- SIS extensions and stability metrics
- synthetic image-style benchmark
- sklearn digits MLP neural-network harness
- downsampled MNIST MLP benchmark suite run

Available as script or future work:

- Fashion-MNIST/CIFAR-10 runs
- small CNN path
- larger multi-example benchmark tables
- adversarial attacks such as FGSM/PGD

Use the saved CSV/JSON files as the source of truth for measured claims.
