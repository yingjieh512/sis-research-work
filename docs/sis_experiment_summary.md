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

Optional dataset arguments exist for `mnist`, `fashion_mnist`, and `cifar10`,
but those require Torch/TorchVision and successful dataset download. Results for
those datasets should be claimed only after running the script and saving
outputs.

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
- CNN, MNIST, Fashion-MNIST, and CIFAR-10 results are not yet measured.
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

Attempt MNIST only after installing Torch/TorchVision:

```bash
python -m experiments.run_nn_sis_experiments --dataset mnist --model mlp --max-examples 1 --train-subset 2000 --test-subset 500
```
