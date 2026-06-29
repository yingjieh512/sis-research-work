# Accelerating and Stabilizing Sufficient Input Subsets for Neural Network Explainability

Yingjie Huang  
Department of Computer Science, University of California, Los Angeles  
yingjieh512@g.ucla.edu

## Abstract

Sufficient Input Subsets (SIS) explain a black-box prediction by identifying a sparse set of observed input features that is sufficient to keep the model confidence above a user-defined threshold. This report documents a research-engineering extension of Google Research's `sufficient_input_subsets` implementation. The project adds SHAP-inspired acceleration, Probabilistic SIS, Hierarchical SIS, and stability metrics for perturbation robustness. The implementation is designed to preserve compatibility with the original `sis_collection` API where possible while adding diagnostics for runtime, `f_batch` calls, individual model evaluations, subset size, sufficiency score, and stability.

The current measured benchmark is a deterministic 8 by 8 synthetic image classifier. On that benchmark, original SIS used 3974 individual model evaluations, while SHAP-guided SIS used 142 individual evaluations with the same final subset size of 7 and the same final sufficiency score of 0.9997. This is a measured 96.43 percent reduction in individual model evaluations for this benchmark. This result supports the resume claim that the framework reduced computational overhead, but it should not be generalized without additional dataset and model experiments. A sklearn digits demo is also runnable and produces qualitative masks, probabilistic heatmaps, and hierarchical explanations. Neural-network experiments on MLPs, LeNet-style CNNs, MNIST, Fashion-MNIST, and CIFAR-10 remain recommended next experiments unless implemented separately.

## 1. Introduction

Interpretability methods are useful when a model's prediction is not enough and a practitioner needs to know which input evidence was necessary for the decision. Sufficient Input Subsets, introduced by Carter et al. [1], provide a local explanation by searching for a subset of features whose observed values alone are sufficient for the same high-confidence decision. Google Research provides a compact NumPy implementation of SIS in the `sufficient_input_subsets` directory [2].

The core advantage of SIS is semantic clarity: an explanation is not just an attribution score, but a subset that satisfies a model-confidence constraint. The practical drawback is cost. Baseline SIS uses iterative backward selection and repeatedly evaluates candidate masked inputs. For images, this can become expensive because even small images contain many possible pixel or region candidates.

This project addresses that cost and broadens the explanatory surface of SIS. It implements:

- SHAP-inspired ranking before SIS search to reduce unnecessary candidate evaluations.
- Probabilistic SIS to estimate uncertainty in explanations across stochastic runs.
- Hierarchical SIS to explain predictions across coarse regions, medium patches, and fine pixels.
- Stability metrics to quantify explanation robustness under small perturbations.

The report is written as a serious technical report rather than as a fake publication. Results are reported only when the code has produced them. Planned experiments and unavailable values are marked explicitly.

## 2. Codebase Inspection Summary

### 2.1 Original Google SIS implementation

The original `sufficient_input_subsets/sis.py` implements SIS with the following public pieces:

- `SISResult`: a namedtuple-like object containing `sis`, `ordering_over_entire_backselect`, `values_over_entire_backselect`, and `mask`.
- `sis_collection(f, threshold, initial_input, fully_masked_input, initial_mask=None)`: finds a disjoint collection of SIS explanations.
- `find_sis(...)`: finds one SIS from a possibly partially masked input.
- `_backselect(...)`: performs backward selection over currently unmasked positions.
- `produce_masked_inputs(...)`: applies boolean masks to produce masked input batches.
- mask helpers such as `make_empty_boolean_mask` and `make_empty_boolean_mask_broadcast_over_axis`.

The original API treats masks as boolean arrays where `True` means the feature is present and `False` means the feature is masked. The mask may have the same shape as the input or may be broadcastable over it, enabling row-level, column-level, channel-level, or region-level masking.

The original tests in `sis_test.py` validate mask construction, backselection behavior, SIS construction, equality behavior, and `sis_collection` outputs on 1D and 2D synthetic examples. In the current workspace, the original test suite passes: 18 tests passed.

### 2.2 Extension code added in this project

The extension files are:

- `shap_guided_sis.py`: implements `shap_guided_sis_collection`, perturbation-based importance estimation, optional SHAP support when available, batched scoring, masked-input caching, ranked SIS construction, pruning, and diagnostics.
- `probabilistic_sis.py`: implements `probabilistic_sis_collection`, `ProbabilisticSISResult`, inclusion probability maps, mean and variance of subset size, confidence summaries, and heatmap helpers.
- `hierarchical_sis.py`: implements `hierarchical_sis_collection`, grid/pixel hierarchy construction, optional superpixel grouping when available, per-level masks, a hierarchy tree, and plotting helpers.
- `stability_metrics.py`: implements `mask_iou`, `mask_f1`, `explanation_stability`, `perturbation_stability`, and `adversarial_sensitivity_score`.
- `benchmark_sis.py`: compares original SIS, SHAP-guided SIS, Probabilistic SIS, and Hierarchical SIS on a deterministic synthetic image problem.
- `vision_demo.py`: runs a small vision demo using sklearn digits if available, with a synthetic fallback.
- `README_extensions.md`: documents the extension APIs and current benchmark results.

The extension test suite passes: 14 tests passed.

### 2.3 Experiments already runnable

The following commands are currently runnable from the repository root:

```bash
python -m unittest discover -s sufficient_input_subsets/tests -p test*.py
python -m sufficient_input_subsets.sis_test
python -m sufficient_input_subsets.benchmark_sis
python -m sufficient_input_subsets.vision_demo --n_probabilistic_samples 3
```

The benchmark writes:

- `sufficient_input_subsets/results/benchmark_report.json`
- `sufficient_input_subsets/results/benchmark_summary.md`

The demo writes:

- `sufficient_input_subsets/results/vision_demo_original.png`
- `sufficient_input_subsets/results/vision_demo_baseline_sis_mask.png`
- `sufficient_input_subsets/results/vision_demo_shap_guided_sis_mask.png`
- `sufficient_input_subsets/results/vision_demo_probabilistic_heatmap.png`
- `sufficient_input_subsets/results/vision_demo_hierarchical_regions.png`
- `sufficient_input_subsets/results/vision_demo_perturbation_stability.png`
- `sufficient_input_subsets/results/vision_demo_summary.json`

### 2.4 Results already available

The current measured synthetic benchmark uses an 8 by 8 deterministic image classifier with threshold 0.75. The initial confidence is 0.999965 and the fully masked confidence is 0.017986.

The current measured sklearn digits demo selected target class 7 with initial confidence 0.999627 and threshold 0.849683. In that demo, baseline SIS selected subset size 8, SHAP-guided SIS selected subset size 8, Probabilistic SIS had mean subset size 8.0, and Hierarchical SIS produced a final subset size of 3. The perturbation demo measured mean explanation drift 0.0222 and mean confidence retention 0.9999.

### 2.5 Claims that still need more evidence

The code supports the resume claims at the implementation level, and the synthetic benchmark supports a measured overhead reduction. However, the following claims need additional benchmark evidence before being stated broadly:

- General "approximately 20 percent" overhead reduction across real neural networks and real datasets.
- Performance on trained MLP/CNN/LeNet/ResNet models.
- MNIST, Fashion-MNIST, or CIFAR-10 results.
- Semantic image region explanations beyond grid regions unless a superpixel dependency and experiment are added.
- Adversarial vulnerability identification under FGSM, PGD, or similar attacks. Current code measures perturbation drift and adversarial sensitivity metrics, but does not run a full adversarial attack benchmark.

## 3. Background: Sufficient Input Subsets

Let `f(x)` be a black-box prediction score, usually the confidence assigned to a target class. Let `x` be the input to explain, `x_masked` be a fully masked baseline input, and `tau` be the confidence threshold. For a boolean mask `m`, the masked input `x_m` keeps features where `m_i = True` and replaces other features with baseline values from `x_masked`.

An SIS is a subset of input features such that:

```text
f(x_m) >= tau
```

and the subset is small or minimal under the search procedure. SIS is a local explanation because the subset explains one prediction for one input, rather than a global property of the model.

Masking strategy matters. In the current implementation, image-like experiments use a zero mask. A stronger future experiment should compare zero masks against mean-value masks because a poor baseline can introduce artifacts.

### 3.1 Baseline SIS pseudocode

```text
Input:
  f_batch: black-box batched scoring function
  x: input to explain
  x_masked: fully masked baseline
  tau: confidence threshold
  M: initial boolean mask, initially all True

Procedure:
  collection = []
  current_input = x
  current_mask = M

  while f_batch(current_input) >= tau:
      stack = []
      mask = current_mask

      while mask has unmasked features:
          candidates = all masks formed by removing one currently present feature
          scores = f_batch(produce_masked_inputs(current_input, x_masked, candidates))
          choose candidate with highest score
          push removed feature and score onto stack
          mask = chosen candidate

      sis = recover smallest suffix of removed features whose restored mask keeps score >= tau
      append SISResult(sis, stack, mask) to collection
      remove this SIS from current_input and current_mask

  return collection
```

This procedure is faithful to the original SIS intuition, but it can require many model evaluations.

## 4. Problem Statement

The project addresses three limitations of vanilla SIS:

1. Computational overhead: Backward selection evaluates many candidate feature removals.
2. Explanation uncertainty: A single SIS can hide the fact that several different subsets are nearly sufficient.
3. Explanation scale: Pixel-level masks can be too granular, while coarse region masks can hide detail.

The goal is not to replace the original SIS implementation. The goal is to extend it while preserving its core API and sufficiency semantics.

## 5. Methodology

### 5.1 Baseline SIS

Baseline SIS is the original `sis_collection` procedure from `sis.py`. It is used as the reference for runtime, model evaluations, subset size, sufficiency score, and threshold satisfaction.

### 5.2 SHAP-guided SIS acceleration

Baseline SIS is expensive because every backward-selection step scores all candidate one-feature removals. The extension adds an importance-ranking stage before SIS search. The public API is:

```python
shap_guided_sis_collection(
    f_batch,
    threshold,
    initial_input,
    fully_masked_input,
    initial_mask=None,
    feature_groups=None,
    importance_method="perturbation",
    max_candidates=None,
    batch_size=64,
    return_diagnostics=True,
)
```

If the optional `shap` package is available, the code can attempt KernelSHAP for singleton feature inputs. If SHAP is unavailable, the default fallback is perturbation-based:

```text
importance(g) = f(x_current) - f(x_current with group g masked)
```

Features or feature groups are ranked by confidence drop. The guided search adds high-importance groups until the threshold is satisfied, then prunes removable groups while preserving sufficiency. The implementation also supports:

- batched scoring through `f_batch`
- memoization of masked inputs
- feature groups for regions rather than individual pixels
- diagnostics for runtime and model evaluations

The overhead reduction percentage is computed as:

```text
100 * (baseline_model_evals - shap_guided_model_evals) / baseline_model_evals
```

Wall-clock runtime reduction is computed as:

```text
100 * (baseline_runtime - shap_guided_runtime) / baseline_runtime
```

The resume wording should be: "Reduced computational overhead by approximately 20 percent in targeted benchmarking; the current synthetic benchmark measured 96.43 percent fewer individual evaluations." If presenting only the current benchmark, say the measured value and dataset/model setting.

### 5.3 Probabilistic SIS

Probabilistic SIS addresses uncertainty. A single deterministic SIS may be unstable when multiple feature subsets are sufficient. The method runs SIS multiple times with stochastic ranking noise or other randomized search choices, then estimates:

```text
P(feature i included) = number of sampled SIS masks containing i / number of samples
```

The output includes:

- sampled SIS results
- inclusion probability map
- mean subset size
- variance of subset size
- confidence summary
- stability summary

This distinguishes features that are consistently necessary from features that appear only because of ranking noise or search instability.

### 5.4 Hierarchical SIS

Hierarchical SIS addresses scale. The algorithm starts from coarse regions and refines selected regions into smaller regions or pixels. A typical hierarchy is:

- Level 1: coarse grid, for example 4 by 4 cells
- Level 2: medium grid, for example 8 by 8 cells
- Level 3: pixel-level or finest available grid

The current benchmark uses levels `(2, 4, 8)` for an 8 by 8 synthetic image. The returned object includes:

- `tree`: per-level selected regions and diagnostics
- `level_masks`: masks at each hierarchy level
- `final_masks`: fine-level masks
- `diagnostics`: runtime-related and configuration metadata

Grid and pixel modes work without heavy dependencies. Optional superpixel grouping is supported only when a lightweight dependency such as `scikit-image` is available.

### 5.5 Stability metrics

The project implements:

1. Mask IoU:

```text
IoU(A, B) = |A intersection B| / |A union B|
```

2. Mask F1:

```text
F1(A, B) = 2TP / (2TP + FP + FN)
```

3. Subset size variance:

```text
Var(|S_1|, ..., |S_k|)
```

4. Confidence retention:

```text
confidence_retention = f(x_SIS_masked) / f(x_original)
```

5. Perturbation stability: average similarity between explanations for clean inputs and slightly perturbed inputs.

6. Adversarial sensitivity score: a combined explanation-drift and confidence-drop summary. The current implementation provides the metric, but full adversarial attack experiments are not yet run.

A robust model should preserve both prediction confidence and explanation structure under small non-semantic perturbations. Large explanation drift can suggest reliance on brittle or adversarially sensitive evidence, but it is a diagnostic signal, not a proof of vulnerability by itself.

## 6. Experimental Setup

### 6.1 Hardware and software

Hardware: local laptop/workstation environment. Exact CPU/GPU information was not captured. Use `[RUN HARDWARE INSPECTION TO FILL THIS VALUE]` if needed.

Software observed in the current environment:

- Python 3.9.13
- NumPy 1.25.2
- scikit-learn 1.3.0
- matplotlib 3.8.2

The original Google requirements are minimal: `absl-py` and NumPy. This workspace includes a small local `absl.testing` compatibility shim so the upstream tests can run without installing `absl-py`.

### 6.2 Datasets

#### Current runnable benchmark: synthetic image classifier

- Input shape: 8 by 8
- Number of classes: binary-style confidence score, not a real class dataset
- Preprocessing: deterministic synthetic pixel intensities
- Masking strategy: zero mask
- Suitability: small, deterministic, fast, and useful for measuring SIS computational overhead
- Limitation: not a real neural-network dataset

#### Current runnable demo: sklearn digits

- Input shape: 8 by 8 grayscale image
- Number of classes: 10 digits
- Train/test split: implemented through `train_test_split` with stratification
- Preprocessing: pixel values scaled by 16.0 to the range `[0, 1]`
- Masking strategy: zero mask
- Current model: logistic regression classifier in `vision_demo.py`
- Suitability: lightweight, local, no dataset download, useful for quick SIS visualization
- Limitation: not currently a neural network in the implementation

#### Recommended primary neural-network dataset: sklearn digits with MLP

- Input shape: 64 flattened features or 1 by 8 by 8 image tensor
- Number of classes: 10
- Train/test split: stratified train/test split
- Preprocessing: scale pixels to `[0, 1]`
- Masking strategy: zero mask or mean pixel mask
- Why suitable: small enough for rapid laptop experiments, but real classification task
- Status: recommended; not currently implemented in benchmark CLI

#### Recommended secondary dataset: MNIST

- Input shape: 1 by 28 by 28
- Number of classes: 10
- Train/test split: standard MNIST train/test split
- Preprocessing: normalize to `[0, 1]` or dataset mean/std
- Masking strategy: zero mask or training-set mean mask
- Model: LeNet-style CNN
- Status: optional future experiment if `torchvision` is installed

#### Optional datasets

Fashion-MNIST can reuse the MNIST pipeline. CIFAR-10 should be optional because SIS over color images is more expensive and the runtime may exceed a normal laptop budget.

### 6.3 Recommended neural-network models

Current code has not yet benchmarked trained neural networks. Recommended models:

#### sklearn digits MLP

- Architecture: input 64, hidden layers 128 and 64, ReLU activations, output 10
- Objective: cross-entropy
- Optimizer: Adam
- Epochs: 20 to 50
- Test accuracy: `[RUN DIGITS MLP TRAINING TO FILL THIS VALUE]`
- SIS threshold: 0.7, 0.8, 0.9
- Why appropriate: fast and defensible for local explainability experiments

#### sklearn digits small CNN

- Architecture: 1 by 8 by 8 input, two small convolutional layers, dense classifier
- Objective: cross-entropy
- Optimizer: Adam
- Epochs: 10 to 30
- Test accuracy: `[RUN DIGITS CNN TRAINING TO FILL THIS VALUE]`
- SIS threshold: 0.8
- Why appropriate: image model with spatial features, but still small

#### MNIST/Fashion-MNIST LeNet-style CNN

- Architecture: Conv-ReLU-Pool, Conv-ReLU-Pool, MLP head
- Objective: cross-entropy
- Optimizer: Adam or SGD with momentum
- Epochs: 3 to 10
- Test accuracy: `[RUN MNIST/FASHION-MNIST EXPERIMENT TO FILL THIS VALUE]`
- SIS threshold: 0.8 or confidence-calibrated threshold

#### CIFAR-10 small CNN or ResNet-18

- Status: optional, runtime-dependent
- Test accuracy: `[RUN CIFAR-10 EXPERIMENT TO FILL THIS VALUE]`
- SIS threshold: 0.7 to 0.9
- Masking strategy: per-channel mean mask preferred over zero mask

### 6.4 Baselines

The primary baseline is original `sis.sis_collection`. Extension variants are:

- SHAP-guided SIS
- Probabilistic SIS
- Hierarchical SIS

Recommended additional ablations:

- no batching
- batching only
- caching only
- SHAP/perturbation guidance only
- batching plus caching plus guidance

### 6.5 Evaluation metrics

The benchmark records:

- wall-clock runtime
- number of `f_batch` calls
- number of individual model evaluations
- subset size
- final confidence on SIS-masked input
- threshold satisfaction
- speedup/reduction against baseline
- stability score

## 7. Results

### 7.1 Table 1: dataset and model summary

| Dataset | Input shape | Model | Test accuracy | Number of evaluated samples | SIS threshold |
| --- | ---: | --- | ---: | ---: | ---: |
| Synthetic image benchmark | 8 by 8 | deterministic synthetic confidence function | not applicable | 1 benchmark input | 0.75 |
| sklearn digits demo | 8 by 8 | logistic regression | `[NOT RECORDED BY CURRENT DEMO]` | 1 high-confidence example | 0.8497 |
| sklearn digits MLP | 64 or 1 by 8 by 8 | recommended MLP | `[RUN BENCHMARK TO FILL THIS VALUE]` | `[RUN BENCHMARK TO FILL THIS VALUE]` | 0.8 recommended |
| MNIST | 1 by 28 by 28 | recommended LeNet-style CNN | `[RUN BENCHMARK TO FILL THIS VALUE]` | `[RUN BENCHMARK TO FILL THIS VALUE]` | 0.8 recommended |

### 7.2 Table 2: runtime and overhead

Measured on the current synthetic image benchmark.

| Method | Runtime (s) | Model evaluations | `f_batch` calls | Subset size | Final confidence | Threshold met | Overhead reduction vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| Original SIS | 0.0051 | 3974 | 128 | 7 | 0.9997 | yes | 0.00% |
| SHAP-guided SIS | 0.0016 | 142 | 19 | 7 | 0.9997 | yes | 96.43% |
| Probabilistic SIS | 0.0081 | 715 | 100 | 8 | 0.9999 | yes | 82.01% |
| Hierarchical SIS | 0.0028 | 129 | 46 | 7 | 0.9997 | yes | 96.75% |

SHAP-guided SIS reduced individual evaluations from 3974 to 142. The formula is:

```text
100 * (3974 - 142) / 3974 = 96.43%
```

Wall-clock runtime reduction for SHAP-guided SIS was:

```text
100 * (0.0051 - 0.0016) / 0.0051 = approximately 69.33%
```

This supports the computational-overhead claim for this benchmark. It does not yet establish the same speedup on MNIST, Fashion-MNIST, CIFAR-10, or trained neural networks.

### 7.3 Table 3: stability metrics

The current benchmark reports a single stability score, and Probabilistic SIS reports pairwise IoU statistics internally. Full F1, confidence-retention, and adversarial-attack tables should be produced in future benchmark extensions.

| Method | Mean IoU under perturbation | Mean F1 under perturbation | Subset size variance | Confidence retention | Adversarial sensitivity score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original SIS | 0.8333 stability score | `[RUN STABILITY BENCHMARK TO FILL THIS VALUE]` | `[RUN STABILITY BENCHMARK TO FILL THIS VALUE]` | `[RUN STABILITY BENCHMARK TO FILL THIS VALUE]` | `[RUN ADVERSARIAL BENCHMARK TO FILL THIS VALUE]` |
| SHAP-guided SIS | 0.7500 stability score | `[RUN STABILITY BENCHMARK TO FILL THIS VALUE]` | `[RUN STABILITY BENCHMARK TO FILL THIS VALUE]` | digits demo: 0.9999 | `[RUN ADVERSARIAL BENCHMARK TO FILL THIS VALUE]` |
| Probabilistic SIS | 0.8333 stability score; sampled mean pairwise IoU 0.8000 | `[RUN STABILITY BENCHMARK TO FILL THIS VALUE]` | synthetic sampled variance 0.0000 | `[RUN STABILITY BENCHMARK TO FILL THIS VALUE]` | `[RUN ADVERSARIAL BENCHMARK TO FILL THIS VALUE]` |
| Hierarchical SIS | 0.7500 stability score | `[RUN STABILITY BENCHMARK TO FILL THIS VALUE]` | `[RUN STABILITY BENCHMARK TO FILL THIS VALUE]` | `[RUN STABILITY BENCHMARK TO FILL THIS VALUE]` | `[RUN ADVERSARIAL BENCHMARK TO FILL THIS VALUE]` |

### 7.4 Table 4: ablation study

The code contains batching, caching, and guidance machinery, but the current benchmark does not run a full ablation grid. The following table is the intended ablation template.

| Variant | Batching | Caching | SHAP guidance | Runtime | Model evaluations | Final confidence |
| --- | :---: | :---: | :---: | ---: | ---: | ---: |
| Original SIS | yes via original batched candidate calls | no extension cache | no | 0.0051 | 3974 | 0.9997 |
| Guidance only | `[IMPLEMENT/RUN ABLATION]` | no | yes | `[RUN ABLATION TO FILL THIS VALUE]` | `[RUN ABLATION TO FILL THIS VALUE]` | `[RUN ABLATION TO FILL THIS VALUE]` |
| Batching + guidance | yes | no | yes | `[RUN ABLATION TO FILL THIS VALUE]` | `[RUN ABLATION TO FILL THIS VALUE]` | `[RUN ABLATION TO FILL THIS VALUE]` |
| Caching + guidance | yes | yes | yes | 0.0016 current SHAP-guided result | 142 | 0.9997 |

## 8. Qualitative Visualizations

The current demo generates the following figures:

Figure 1: Original input image and predicted label. This shows the image selected by the demo and provides context for all explanation masks.

Figure 2: Baseline SIS mask. This shows which pixels/features original SIS selected as sufficient.

Figure 3: SHAP-guided SIS mask. This should be compared with Figure 2 to see whether acceleration preserves a similar sufficient rationale.

Figure 4: Probabilistic inclusion heatmap. Bright regions indicate features that are included frequently across stochastic SIS samples.

Figure 5: Hierarchical SIS explanation at coarse, medium, and fine levels. This shows how the explanation changes as the feature grouping becomes more granular.

Figure 6: Stability under perturbation plot. This summarizes explanation drift and confidence retention under small perturbations.

Generated artifact paths are under `sufficient_input_subsets/results/`.

## 9. Discussion

The main measured result is that SHAP-guided SIS substantially reduces individual model evaluations on the synthetic benchmark while preserving threshold satisfaction and final sufficiency score. The result is stronger than the target 20 percent reduction, but it is measured on a small deterministic benchmark. The honest explanation is that the implementation has demonstrated a large reduction in a controlled benchmark and is designed to measure the same quantities on larger datasets and neural models.

Probabilistic SIS adds value because it turns explanation variability into a measurable object. Instead of presenting one mask as definitive, it reports inclusion probabilities and subset-size statistics. This is important because SIS can have multiple valid sufficient subsets.

Hierarchical SIS adds value because it makes SIS explanations easier to inspect at multiple scales. Coarse grid regions give a high-level view, while finer masks show which pixels or patches preserve model confidence.

Stability metrics connect SIS to robustness analysis. A model can preserve a class prediction while changing its explanation under small perturbations. That drift can be a useful warning sign. However, current code does not prove adversarial vulnerability. It provides baseline metrics and hooks for future adversarial experiments.

## 10. Limitations

- The current measured benchmark is synthetic, not a trained neural network.
- The current sklearn digits demo uses logistic regression, not an MLP or CNN.
- MNIST, Fashion-MNIST, and CIFAR-10 experiments are not yet implemented in the benchmark CLI.
- Exact SHAP is optional; the default method is SHAP-inspired perturbation ranking.
- Grid hierarchy is not the same as true semantic segmentation.
- Full adversarial experiments such as FGSM or PGD have not been run.
- Runtime measurements are local and small-scale; they should be rerun on the final interview machine if exact numbers matter.

## 11. Future Work

- Add CLI support for `--dataset digits --model mlp --n-samples`.
- Train and evaluate a digits MLP and small CNN.
- Add optional PyTorch/TorchVision support for MNIST and Fashion-MNIST.
- Add CIFAR-10 only as an optional longer-running experiment.
- Implement an ablation runner for batching, caching, and guidance.
- Add FGSM/PGD perturbation experiments for adversarial sensitivity.
- Compare zero masking against mean-value masking.
- Add superpixel hierarchy experiments when `scikit-image` is available.

## 12. Conclusion

This project extends SIS from a compact black-box explanation algorithm into a practical research-engineering framework. The implemented system preserves the original SIS API where possible, adds SHAP-inspired acceleration, supports probabilistic and hierarchical explanations, and measures explanation stability. The current benchmark shows a measured 96.43 percent reduction in individual model evaluations on a deterministic synthetic image task, while the digits demo verifies that the visualization path is runnable. The strongest defensible claim is that the implementation provides the machinery and benchmark evidence for efficient SIS on small image-style tasks, with broader neural-network claims requiring the additional experiments listed above.

## Appendix A. Interview Explanation

### 30-second explanation

I extended Google Research's Sufficient Input Subsets implementation. SIS finds the smallest set of input features that keeps a model's prediction above a confidence threshold. My project adds a SHAP-inspired ranking step to reduce model evaluations, a probabilistic version to estimate uncertainty in explanations, a hierarchical version for coarse-to-fine image masks, and stability metrics for perturbation robustness. In the current synthetic benchmark, SHAP-guided SIS reduced individual model evaluations from 3974 to 142 while preserving sufficiency.

### 60-second explanation

The original SIS algorithm is faithful but expensive because it repeatedly evaluates many masked versions of an input. I kept the original `sis_collection` result format and built wrappers around it. The SHAP-guided version estimates feature importance first using either optional SHAP or a perturbation fallback that masks each feature or group and measures the confidence drop. It then prioritizes high-importance features, batches model calls, caches masked evaluations, and prunes the final subset. I also added Probabilistic SIS, which samples multiple plausible explanations and reports inclusion probabilities, and Hierarchical SIS, which starts with coarse regions and refines to pixels. Stability metrics quantify whether explanations remain similar under small perturbations.

### Do not overclaim

Say: "The current synthetic benchmark measured a 96.43 percent reduction in individual model evaluations."  
Do not say: "This always reduces overhead by 96 percent."  
Say: "The project provides a baseline for adversarial vulnerability analysis through explanation drift metrics."  
Do not say: "The project proves adversarial vulnerabilities in deep models."

## Appendix B. Commands to Reproduce Current Experiments

```bash
python -m unittest discover -s sufficient_input_subsets/tests -p test*.py
python -m sufficient_input_subsets.sis_test
python -m sufficient_input_subsets.benchmark_sis
python -m sufficient_input_subsets.vision_demo --n_probabilistic_samples 3
```

The commands requested in the original project plan below are not currently implemented by the CLI:

```bash
python -m pytest tests/
python vision_demo.py --dataset digits --model mlp --n-samples 20
python benchmark_sis.py --dataset digits --model mlp --threshold 0.8 --n-samples 50
python benchmark_sis.py --dataset mnist --model lenet --threshold 0.8 --n-samples 50
```

To make these commands work, add argparse flags for dataset/model/sample count, implement MLP/CNN training pipelines, and optionally add PyTorch/TorchVision dependencies for MNIST.

## Appendix C. References

[1] Brandon Carter, Jonas Mueller, Siddhartha Jain, and David K. Gifford. "What made you do this? Understanding black-box decisions with sufficient input subsets." arXiv:1810.03805, 2018. https://arxiv.org/abs/1810.03805

[2] Google Research Authors. `sufficient_input_subsets` implementation in `google-research`. https://github.com/google-research/google-research/tree/master/sufficient_input_subsets

[3] Scott M. Lundberg and Su-In Lee. "A Unified Approach to Interpreting Model Predictions." Advances in Neural Information Processing Systems 30, 2017. https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html

[4] Fabian Pedregosa et al. "Scikit-learn: Machine Learning in Python." Journal of Machine Learning Research, 12:2825-2830, 2011. https://jmlr.org/papers/v12/pedregosa11a.html

[5] Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy. "Explaining and Harnessing Adversarial Examples." arXiv:1412.6572, 2014. https://arxiv.org/abs/1412.6572

## Appendix D. Addendum: Measured Digits MLP Neural-Network Harness

After the initial report draft, the repository added `experiments/run_nn_sis_experiments.py`, a CPU-friendly neural-network experiment harness. The harness trains a sklearn `MLPClassifier` on the sklearn digits dataset and exposes the model through an SIS-compatible black-box scoring function.

Command:

```bash
python -m experiments.run_nn_sis_experiments --dataset digits --model mlp --max-examples 1 --seed 0 --output-dir results/sis_nn_experiments
```

Measured output files:

- `results/sis_nn_experiments/results.csv`
- `results/sis_nn_experiments/results.json`

Measured run metadata:

- Dataset: sklearn digits, 8 by 8 grayscale
- Model: sklearn MLPClassifier
- Train accuracy: 0.9911
- Test accuracy: 0.9733
- Selected example: test index 187, target class 4
- Original confidence: 0.9999985
- SIS threshold: 0.8

| Method | Threshold met | Final confidence | Subset size | Evaluations | f_batch calls | Runtime (s) | Eval reduction | Stability IoU | Stability F1 |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original SIS | yes | 0.999998 | 4 | 7945 | 255 | 0.0432 | 0.00% | 0.6250 | 0.7000 |
| SHAP-guided SIS | yes | 0.998905 | 3 | 199 | 13 | 0.0035 | 97.50% | 0.5417 | 0.6786 |
| Probabilistic SIS | yes | 0.998905 | 3 | 600 | 42 | 0.0114 | 92.45% | 0.5417 | 0.6786 |
| Hierarchical SIS | yes | 0.963973 | 2 | 40 | 23 | 0.0040 | 99.50% | 0.6667 | 0.7500 |

This addendum supersedes earlier statements that no neural-network experiment had been measured. The broader caveats still apply: this is one lightweight sklearn digits MLP run, not MNIST/CIFAR/CNN evidence.

## Appendix E. Addendum: Measured Digits + MNIST Benchmark Suite

The repository now includes `experiments/run_nn_benchmark_suite.py`, which runs the neural-network SIS harness over multiple datasets and writes aggregate CSV, JSON, and Markdown reports. The suite records unavailable datasets explicitly rather than inventing missing results.

Command:

```bash
python -m experiments.run_nn_benchmark_suite --datasets digits,mnist --max-examples 1 --max-iter 80 --mnist-train-subset 2000 --mnist-test-subset 500 --mnist-image-size 14 --stability-perturbations 0 --output-dir results/sis_nn_benchmarks
```

Measured output directory:

- `results/sis_nn_benchmarks/benchmark_suite_20260628_233409/`

Benchmark setup:

- digits: sklearn digits, 8 by 8 grayscale, sklearn MLPClassifier, test accuracy 0.9733.
- MNIST: OpenML/TorchVision MNIST path, 28 by 28 images average-pooled to 14 by 14, 2,000 training samples, 500 test samples, sklearn MLPClassifier, test accuracy 0.8940.
- Evaluation: one selected high-confidence test example per dataset, threshold 0.8, zero masking, no perturbation-stability probes in this suite run.

| Dataset | Method | Threshold met | Final confidence | Subset size | Evaluations | f_batch calls | Runtime (s) | Evaluation reduction |
| --- | --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| digits | Original SIS | yes | 0.999998 | 4 | 7945 | 255 | 0.0506 | 0.00% |
| digits | SHAP-guided SIS | yes | 0.998905 | 3 | 199 | 13 | 0.0036 | 97.50% |
| digits | Probabilistic SIS | yes | 0.998905 | 3 | 600 | 42 | 0.0113 | 92.45% |
| digits | Hierarchical SIS | yes | 0.963973 | 2 | 40 | 23 | 0.0043 | 99.50% |
| MNIST 14x14 | Original SIS | yes | 0.998483 | 10 | 56370 | 584 | 0.2341 | 0.00% |
| MNIST 14x14 | SHAP-guided SIS | yes | 0.994795 | 6 | 405 | 23 | 0.0098 | 99.28% |
| MNIST 14x14 | Probabilistic SIS | yes | 0.994795 | 6 | 1218 | 72 | 0.0280 | 97.84% |
| MNIST 14x14 | Hierarchical SIS | yes | 0.821547 | 2 | 121 | 28 | 0.0093 | 99.79% |

This addendum supersedes earlier statements that MNIST had not been benchmarked at all. The measured MNIST result is still deliberately modest: it is a one-example, downsampled, MLP-based benchmark rather than a full MNIST CNN study. It should not be described as CIFAR-10 evidence, adversarial evidence, or broad deep-vision evidence.
