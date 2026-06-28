# Reproducibility Checklist

## Current workspace status

- Repository root: `C:\Users\Yingjie Huang\Downloads\sis-research-work`
- Python: 3.9.13
- NumPy: 1.25.2
- scikit-learn: 1.3.0
- matplotlib: 3.8.2
- Hardware: `[RUN HARDWARE INSPECTION TO FILL THIS VALUE]`

## Current tests

Run from the repository root:

```bash
python -m unittest discover -s sufficient_input_subsets/tests -p test*.py
python -m sufficient_input_subsets.sis_test
```

Latest observed result:

- Extension tests: 14 tests passed.
- Original Google SIS tests: 18 tests passed.

## Current benchmark command

```bash
python -m sufficient_input_subsets.benchmark_sis
```

Outputs:

- `sufficient_input_subsets/results/benchmark_report.json`
- `sufficient_input_subsets/results/benchmark_summary.md`

Latest observed synthetic benchmark:

| Method | Runtime (s) | Individual evals | `f_batch` calls | Subset size | Final confidence | Threshold met | Reduction |
| --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| Original SIS | 0.0051 | 3974 | 128 | 7 | 0.9997 | yes | 0.00% |
| SHAP-guided SIS | 0.0016 | 142 | 19 | 7 | 0.9997 | yes | 96.43% |
| Probabilistic SIS | 0.0081 | 715 | 100 | 8 | 0.9999 | yes | 82.01% |
| Hierarchical SIS | 0.0028 | 129 | 46 | 7 | 0.9997 | yes | 96.75% |

## Current vision demo command

```bash
python -m sufficient_input_subsets.vision_demo --n_probabilistic_samples 3
```

Outputs:

- `sufficient_input_subsets/results/vision_demo_original.png`
- `sufficient_input_subsets/results/vision_demo_baseline_sis_mask.png`
- `sufficient_input_subsets/results/vision_demo_shap_guided_sis_mask.png`
- `sufficient_input_subsets/results/vision_demo_probabilistic_heatmap.png`
- `sufficient_input_subsets/results/vision_demo_hierarchical_regions.png`
- `sufficient_input_subsets/results/vision_demo_perturbation_stability.png`
- `sufficient_input_subsets/results/vision_demo_summary.json`

Latest observed demo:

- Dataset path: `sklearn_digits`
- Current model: logistic regression
- Target class: 7
- Initial confidence: 0.999627
- SIS threshold: 0.849683
- Baseline subset size: 8
- SHAP-guided subset size: 8
- Probabilistic mean subset size: 8.0
- Hierarchical final subset size: 3
- Mean explanation drift under perturbation: 0.0222
- Mean confidence retention: 0.9999

## Commands requested in the target paper plan

The following commands are useful target commands, but they do not currently work exactly as written because the CLI does not yet support `--dataset`, `--model`, and `--n-samples` for those scripts:

```bash
python -m pytest tests/
python vision_demo.py --dataset digits --model mlp --n-samples 20
python benchmark_sis.py --dataset digits --model mlp --threshold 0.8 --n-samples 50
python benchmark_sis.py --dataset mnist --model lenet --threshold 0.8 --n-samples 50
```

To make them work, implement:

- `argparse` support for `--dataset`, `--model`, `--n-samples`, and `--threshold`.
- A digits MLP training/evaluation path.
- A digits CNN path if PyTorch or scikit-learn-compatible image modeling is chosen.
- Optional PyTorch/TorchVision support for MNIST and Fashion-MNIST.
- A benchmark loop over multiple correctly classified high-confidence samples.
- Result aggregation across samples.

## Recommended reproducible experiment matrix

### Digits MLP

- Dataset: sklearn digits
- Model: MLP with hidden layers 128 and 64
- Epochs: 20 to 50
- Optimizer: Adam
- Mask: zero mask and mean mask
- Thresholds: 0.7, 0.8, 0.9
- Samples: 20, then 50
- Runs: 3 random seeds
- Required output: runtime, model evaluations, subset size, sufficiency, stability

### Digits CNN

- Dataset: sklearn digits
- Model: small CNN over 1 by 8 by 8 inputs
- Threshold: 0.8
- Samples: 20 to 50
- Required output: same as digits MLP

### MNIST LeNet

- Dataset: MNIST
- Dependency: `torch`, `torchvision`
- Model: LeNet-style CNN
- Threshold: 0.8
- Samples: 50
- Required output: same as digits MLP
- Status: not currently implemented

### Adversarial sensitivity

- Perturbations: Gaussian noise, random pixel dropout, FGSM, PGD
- Metrics: clean IoU, perturbed IoU, F1, subset-size variance, confidence retention, adversarial sensitivity score
- Status: perturbation hooks exist; FGSM/PGD experiments not currently implemented

## Minimum checklist before making broad resume claims

- [x] Original SIS tests pass.
- [x] Extension tests pass.
- [x] Synthetic benchmark reports runtime and model-evaluation counts.
- [x] SHAP-guided overhead reduction is measured on at least one benchmark.
- [x] Probabilistic and Hierarchical SIS are implemented.
- [x] Stability metrics are implemented.
- [ ] Digits MLP/CNN benchmark is implemented and run.
- [ ] MNIST or Fashion-MNIST benchmark is implemented and run.
- [ ] Ablation table is generated.
- [ ] FGSM/PGD adversarial sensitivity experiment is implemented and run.
- [ ] Hardware details are recorded.

## Honest interpretation rule

Use "implemented" for code that exists and passes tests.

Use "measured" only for values present in `benchmark_report.json` or `vision_demo_summary.json`.

Use "planned", "recommended", or `[RUN BENCHMARK TO FILL THIS VALUE]` for future experiments.
