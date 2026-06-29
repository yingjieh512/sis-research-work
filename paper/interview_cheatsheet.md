# SIS Extensions Interview Cheatsheet

## 30-second explanation

I extended Google Research's Sufficient Input Subsets implementation. SIS finds a minimal subset of input features that keeps a model prediction above a confidence threshold. My extensions add SHAP-inspired acceleration to reduce model evaluations, Probabilistic SIS to estimate uncertainty in explanations, Hierarchical SIS for coarse-to-fine image explanations, and stability metrics to measure explanation drift under perturbations. In the current synthetic benchmark, SHAP-guided SIS reduced individual model evaluations from 3974 to 142 while preserving the final sufficiency score.

## 60-second explanation

The original SIS algorithm is faithful but computationally expensive because it repeatedly evaluates masked versions of the input during backward selection. I kept the original `sis_collection` and `SISResult` semantics, then built wrappers around them. The SHAP-guided version first estimates feature importance using either optional SHAP or a perturbation fallback: mask each feature or region, measure the drop in target confidence, rank features by that drop, and prioritize high-ranked features during SIS search. I also added batching and caching so repeated masked evaluations are avoided. Probabilistic SIS repeats the search with noisy rankings and reports inclusion probabilities, while Hierarchical SIS starts with coarse regions and refines to pixels. Stability metrics quantify whether explanations remain similar under small input perturbations.

## 2-minute technical explanation

SIS takes a black-box scoring function `f(x)`, an input `x`, a fully masked baseline `x_masked`, and a threshold `tau`. A subset is sufficient if the model score on the masked input remains above threshold: `f(x_S) >= tau`. The original Google implementation performs backward selection: it repeatedly asks which single feature can be removed while preserving the highest score, then reconstructs a sufficient subset from the backselection stack. That gives a clear local explanation, but it can be expensive because each step evaluates many candidates.

My SHAP-guided SIS changes the search order. Before constructing the subset, it estimates feature or region importance. If real SHAP is installed, the code can optionally use it on small singleton-feature inputs. Otherwise it uses a SHAP-inspired perturbation approximation: mask a feature or group, run `f_batch`, and measure the confidence drop. The implementation batches these calls and memoizes masked inputs. Then it adds high-ranked groups until the threshold is met and prunes unnecessary groups. Diagnostics report runtime, batched calls, individual evaluations, subset size, sufficiency score, and speedup against the baseline.

The probabilistic extension samples multiple SIS explanations with ranking noise and aggregates feature inclusion probabilities. This helps when several subsets are sufficient. The hierarchical extension runs SIS over grid regions at progressively finer levels, producing coarse, medium, and fine masks. Stability metrics such as IoU, F1, subset-size variance, confidence retention, perturbation drift, and adversarial sensitivity provide a way to quantify how robust the explanation is under small perturbations.

## Likely interviewer questions

### What is SIS?

Sufficient Input Subsets is a local interpretability method. It identifies a subset of input features whose observed values are enough to keep the model prediction above a confidence threshold, even when all other features are masked.

### What was your contribution?

I built a research-engineering extension around the original Google SIS implementation. I added SHAP-inspired acceleration, probabilistic explanation sampling, hierarchical multi-scale explanation masks, stability metrics, unit tests, a benchmark runner, a vision demo, and documentation.

### What dataset did you use?

The current measured benchmark uses a deterministic 8 by 8 synthetic image classifier to measure computational overhead. The runnable vision demo uses the sklearn digits dataset when available. The recommended next datasets are sklearn digits with an MLP or small CNN, MNIST with a LeNet-style CNN, optional Fashion-MNIST, and optional CIFAR-10.

### What model did you use?

The measured benchmark uses a deterministic synthetic confidence function. The current digits demo uses logistic regression. Neural-network experiments are recommended but not yet measured in the current benchmark. I should not claim completed MLP/CNN results until those experiments are added and run.

### How did you measure the 20% speedup?

The benchmark wraps `f_batch` in a counting function and records both batched function calls and individual model evaluations. The formula is:

```text
100 * (baseline_model_evals - shap_guided_model_evals) / baseline_model_evals
```

On the current synthetic benchmark, original SIS used 3974 individual model evaluations and SHAP-guided SIS used 142, which is a 96.43 percent reduction. The resume phrase "approximately 20 percent" is defensible as a conservative statement only if I explain the measured benchmark setting.

### What is SHAP-guided acceleration?

It is a feature-ranking layer before SIS search. If SHAP is installed, the implementation can optionally use SHAP values for small singleton-feature inputs. Otherwise it uses a perturbation method: mask each feature or group, measure the drop in confidence, rank by confidence drop, and prioritize those features during SIS construction and pruning.

### What is Probabilistic SIS?

Probabilistic SIS repeats SIS multiple times with stochastic ranking perturbations. It computes an inclusion probability for each feature, mean subset size, subset-size variance, and stability summaries. This shows which features are consistently selected versus which are search-dependent.

### What is Hierarchical SIS?

Hierarchical SIS runs SIS at multiple spatial scales. It starts with coarse grid regions, refines selected regions into smaller patches, and can end at pixel level. This gives explanations that are easier to inspect than raw pixel masks alone.

### What are stability metrics?

They measure how similar explanation masks remain under repeated runs or input perturbations. Implemented metrics include IoU, F1, subset-size variance, confidence retention, perturbation stability, and adversarial sensitivity score.

### How does this relate to adversarial robustness?

If small input perturbations cause large explanation drift, that can suggest the model relies on brittle evidence. The current project provides metrics and perturbation hooks for this analysis. It does not yet prove adversarial vulnerability because full adversarial attack experiments such as FGSM or PGD have not been run.

### What would you improve next?

I would add a full digits MLP/CNN benchmark, then MNIST with a LeNet-style CNN. I would add ablations for batching, caching, and guidance, compare zero masks with mean masks, and run FGSM/PGD perturbation studies to connect explanation drift more directly to adversarial sensitivity.

## Do not overclaim

Implemented:

- SHAP-inspired SIS wrapper with batching, caching, diagnostics, and optional SHAP path.
- Probabilistic SIS with inclusion probabilities.
- Hierarchical SIS with grid and pixel modes.
- Stability metrics.
- Unit tests, synthetic benchmark, and sklearn digits demo.

Measured:

- Extension tests: 14/14 passed.
- Original SIS tests: 18/18 passed.
- Synthetic benchmark: SHAP-guided SIS reduced individual evaluations from 3974 to 142, or 96.43 percent.
- Digits demo: runnable qualitative path with logistic regression; target class 7; initial confidence 0.9996; threshold 0.8497; mean perturbation explanation drift 0.0222.

Planned or not yet measured:

- MLP/CNN neural-network benchmark.
- MNIST/Fashion-MNIST/CIFAR-10 experiments.
- Full ablation table.
- Full adversarial attack experiments.
- Semantic superpixel hierarchy experiments.

Safe resume wording:

"Developed a SIS-based interpretability framework extending Google Research's implementation with SHAP-inspired acceleration, probabilistic explanation sampling, hierarchical image masks, and stability metrics; measured a 96.43 percent reduction in individual model evaluations on a controlled synthetic benchmark and designed the framework to benchmark overhead reduction on real vision models."

More conservative resume wording:

"Developed a SIS-based interpretability framework with SHAP-inspired feature ranking, probabilistic and hierarchical explanation variants, and stability metrics; benchmarked reduced model-evaluation overhead on small image-style tasks while preserving sufficiency."

## Addendum: MNIST benchmark wording

A small MNIST benchmark is now measured through `experiments/run_nn_benchmark_suite.py`. The measured run used MNIST loaded via TorchVision/OpenML, average-pooled images to 14 by 14, trained a sklearn MLP on 2,000 examples, evaluated 500 test examples, and explained one selected high-confidence example. SHAP-guided SIS reduced individual evaluations from 56,370 to 405 on that selected example while preserving the 0.8 threshold. This is a downsampled MLP benchmark, not a full MNIST CNN claim.
