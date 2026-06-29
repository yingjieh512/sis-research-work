# SIS Experiment Summary

## Repository inspection

The repository contains:

- `sufficient_input_subsets/sis.py`: original Google-style SIS implementation.
- `sufficient_input_subsets/shap_guided_sis.py`: perturbation/SHAP-inspired guided SIS with batching, caching, and diagnostics.
- `sufficient_input_subsets/probabilistic_sis.py`: stochastic SIS sampling and inclusion probabilities.
- `sufficient_input_subsets/hierarchical_sis.py`: grid/pixel hierarchy explanations.
- `sufficient_input_subsets/stability_metrics.py`: IoU, F1, drift, confidence-retention, and adversarial-sensitivity helpers.
- `sufficient_input_subsets/benchmark_sis.py`: deterministic synthetic image benchmark.
- `sufficient_input_subsets/vision_demo.py`: sklearn digits demo with synthetic fallback.
- `experiments/run_nn_sis_experiments.py`: CPU-friendly neural-network experiment harness.
- `experiments/run_nn_benchmark_suite.py`: aggregate digits/MNIST benchmark-suite runner.
- `paper/`: technical report, LaTeX source, interview cheatsheet, reproducibility checklist, and resume wording notes.

The original SIS API is preserved where practical. The extension methods return
SIS-compatible masks/collections plus diagnostics rather than changing
`sis_collection`.

## Neural-network harness

Run:

```bash
python -m experiments.run_nn_sis_experiments --dataset digits --model mlp --max-examples 1 --seed 0 --output-dir results/sis_nn_experiments
```

Default behavior:

- Dataset: sklearn digits
- Input shape: 8x8 grayscale
- Preprocessing: scale pixels to `[0, 1]`
- Split: stratified train/test split
- Model: sklearn `MLPClassifier`
- Mask baseline: zero image
- Example selection: correctly classified examples with confidence at least `0.70`
- Threshold mode: relative by default, with default threshold cap `0.8`
- Stability: Gaussian noise with standard deviation `0.02`, two perturbations by default

MNIST can be benchmarked with either TorchVision or a scikit-learn OpenML
fallback. The benchmark suite defaults to 14x14 average-pooled MNIST to keep
baseline SIS CPU-friendly. Fashion-MNIST and CIFAR-10 still require
Torch/TorchVision and should be claimed only after their runs complete.

## Current measured neural-network results

Output files:

- `results/sis_nn_experiments/results.csv`
- `results/sis_nn_experiments/results.json`

Run metadata:

- Dataset: `digits`
- Model: `mlp`
- Model library: `sklearn.neural_network.MLPClassifier`
- Train accuracy: `0.9911`
- Test accuracy: `0.9733`
- Selected examples: `1`
- Selected test index: `187`
- Target class: `4`
- Original confidence: `0.9999985`
- Threshold: `0.8`

| Method | Threshold met | Final confidence | Subset size | Evaluations | f_batch calls | Runtime (s) | Eval reduction | Stability IoU | Stability F1 |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original SIS | yes | 0.999998 | 4 | 7945 | 255 | 0.0432 | 0.00% | 0.6250 | 0.7000 |
| SHAP-guided SIS | yes | 0.998905 | 3 | 199 | 13 | 0.0035 | 97.50% | 0.5417 | 0.6786 |
| Probabilistic SIS | yes | 0.998905 | 3 | 600 | 42 | 0.0114 | 92.45% | 0.5417 | 0.6786 |
| Hierarchical SIS | yes | 0.963973 | 2 | 40 | 23 | 0.0040 | 99.50% | 0.6667 | 0.7500 |

Interpretation:

- The digits MLP run confirms the project can explain a real neural-network model through an SIS-compatible black-box scoring wrapper.
- SHAP-guided SIS reduced individual evaluations from `7945` to `199` on this selected example while keeping confidence above threshold.
- Probabilistic and Hierarchical SIS also satisfied the threshold in this run.
- These are single-example lightweight results, not broad dataset-level claims.

## Measured digits + MNIST benchmark suite

Command:

```bash
python -m experiments.run_nn_benchmark_suite --datasets digits,mnist --max-examples 1 --max-iter 80 --mnist-train-subset 2000 --mnist-test-subset 500 --mnist-image-size 14 --stability-perturbations 0 --output-dir results/sis_nn_benchmarks
```

Output directory:

- `results/sis_nn_benchmarks/benchmark_suite_20260628_233409/`

| Dataset | Test accuracy | Method | Threshold met | Final confidence | Subset size | Evaluations | Runtime (s) | Eval reduction |
| --- | ---: | --- | :---: | ---: | ---: | ---: | ---: | ---: |
| digits | 0.9733 | Original SIS | yes | 0.999998 | 4 | 7945 | 0.0506 | 0.00% |
| digits | 0.9733 | SHAP-guided SIS | yes | 0.998905 | 3 | 199 | 0.0036 | 97.50% |
| digits | 0.9733 | Probabilistic SIS | yes | 0.998905 | 3 | 600 | 0.0113 | 92.45% |
| digits | 0.9733 | Hierarchical SIS | yes | 0.963973 | 2 | 40 | 0.0043 | 99.50% |
| MNIST 14x14 | 0.8940 | Original SIS | yes | 0.998483 | 10 | 56370 | 0.2341 | 0.00% |
| MNIST 14x14 | 0.8940 | SHAP-guided SIS | yes | 0.994795 | 6 | 405 | 0.0098 | 99.28% |
| MNIST 14x14 | 0.8940 | Probabilistic SIS | yes | 0.994795 | 6 | 1218 | 0.0280 | 97.84% |
| MNIST 14x14 | 0.8940 | Hierarchical SIS | yes | 0.821547 | 2 | 121 | 0.0093 | 99.79% |

Interpretation: the MNIST benchmark is measured, but it is intentionally small:
one selected example, an MLP, 2,000 training samples, 500 test samples, and
14x14 downsampling. It is not evidence for full-scale MNIST CNN or CIFAR-10
behavior.

## Existing synthetic benchmark

The synthetic benchmark remains available:

```bash
python -m sufficient_input_subsets.benchmark_sis
```

It writes:

- `sufficient_input_subsets/results/benchmark_report.json`
- `sufficient_input_subsets/results/benchmark_summary.md`

That benchmark is useful for controlled overhead measurement but should be
described as synthetic rather than as a neural-network result.

## Limitations

- The checked neural-network result is one high-confidence sklearn digits MLP example.
- A small downsampled MNIST MLP benchmark is measured; CNN, Fashion-MNIST, and CIFAR-10 results are not yet measured.
- The `small_cnn` option is reserved for a future PyTorch path.
- Stability is measured with small Gaussian perturbations, not a full adversarial attack.
- Larger multi-example benchmarks should be run before making broad claims.

## Recommended next commands

Run a slightly larger digits MLP experiment:

```bash
python -m experiments.run_nn_sis_experiments --dataset digits --model mlp --max-examples 5 --seed 0
```

Skip slower stochastic/hierarchical methods for a faster baseline-vs-guided pass:

```bash
python -m experiments.run_nn_sis_experiments --dataset digits --model mlp --max-examples 10 --skip-probabilistic --skip-hierarchical --stability-perturbations 0
```

Run the measured MNIST benchmark path using TorchVision or the OpenML fallback:

```bash
python -m experiments.run_nn_benchmark_suite --datasets mnist --max-examples 1 --mnist-train-subset 2000 --mnist-test-subset 500 --mnist-image-size 14
```
