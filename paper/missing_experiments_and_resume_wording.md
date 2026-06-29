# Missing Experiments and Resume Wording

## Missing experiments or data

These are the main gaps before making broad claims about deep vision models:

1. Digits neural-network benchmark
   - Implement an MLP or small CNN for sklearn digits.
   - Report train/test split, test accuracy, threshold, number of evaluated samples, runtime, model evaluations, subset size, sufficiency, and stability.

2. MNIST or Fashion-MNIST benchmark
   - Add optional PyTorch/TorchVision dependency.
   - Train a LeNet-style CNN.
   - Evaluate original SIS, SHAP-guided SIS, Probabilistic SIS, and Hierarchical SIS across multiple samples.

3. CIFAR-10 benchmark
   - Optional only if runtime is acceptable.
   - Prefer a small CNN before trying ResNet-18.
   - Use a per-channel mean mask instead of a zero mask.

4. Full ablation table
   - Compare original SIS, batching only, caching only, guidance only, and batching plus caching plus guidance.
   - Measure runtime, `f_batch` calls, individual model evaluations, final confidence, and threshold satisfaction.

5. Full stability benchmark
   - Compute IoU, F1, subset-size variance, confidence retention, and explanation drift across multiple samples and perturbation types.

6. Adversarial sensitivity benchmark
   - Implement FGSM and PGD or another clearly defined adversarial perturbation.
   - Report explanation drift and confidence changes.
   - Do not claim adversarial vulnerabilities are proven until this is done.

7. Hardware and reproducibility metadata
   - Record CPU, GPU, RAM, OS, Python version, library versions, random seeds, and exact commands.

## Results that are currently measured

Current synthetic benchmark:

- Original SIS individual model evaluations: 3974
- SHAP-guided SIS individual model evaluations: 142
- Measured reduction: 96.43 percent
- Original SIS final confidence: 0.9997
- SHAP-guided SIS final confidence: 0.9997
- Original SIS subset size: 7
- SHAP-guided SIS subset size: 7

Current tests:

- Extension tests: 14/14 passed
- Original SIS tests: 18/18 passed

Current demo:

- Dataset path: sklearn digits
- Current model: logistic regression
- Target class: 7
- Initial confidence: 0.9996
- Threshold: 0.8497
- Baseline subset size: 8
- SHAP-guided subset size: 8
- Mean perturbation explanation drift: 0.0222
- Mean confidence retention: 0.9999

## Suggested resume wording

### Strong but honest version

Developed a SIS-based interpretability framework extending Google Research's implementation with SHAP-inspired feature ranking, probabilistic explanation sampling, hierarchical image masks, and explanation-stability metrics; benchmarked a 96.43 percent reduction in individual model evaluations on a controlled synthetic image task while preserving SIS sufficiency.

### Conservative version

Developed a high-performance interpretability framework based on Sufficient Input Subsets, adding SHAP-inspired acceleration, Probabilistic SIS, Hierarchical SIS, and stability metrics; validated the implementation with unit tests, a synthetic overhead benchmark, and a lightweight sklearn digits visualization demo.

### If you want to keep the "approximately 20 percent" claim

Developed a SIS-based interpretability framework and reduced feature-selection overhead through SHAP-inspired ranking, batching, and caching; current controlled benchmarks exceed the target overhead reduction, with additional neural-network experiments planned for broader validation.

### Avoid this wording for now

Avoid:

"Reduced computational overhead by 20 percent on deep vision models."

Reason:

The current measured reduction is on a synthetic image benchmark, not a trained deep vision model.

Avoid:

"Identified adversarial vulnerabilities in deep vision models."

Reason:

The code implements stability and adversarial sensitivity metrics, but full adversarial attack experiments have not yet been run.

Better:

"Implemented SIS stability metrics that provide a baseline for studying explanation drift and potential adversarial sensitivity."

## Interview-safe phrasing

If asked whether the project used neural networks:

"The framework is designed for black-box neural networks because SIS only needs `f_batch`. The current committed benchmark uses a deterministic synthetic image classifier to measure overhead, and the runnable digits demo uses logistic regression for speed. The next validation step is to plug in a digits MLP/CNN and MNIST LeNet, which the report outlines."

If asked whether the 20 percent speedup is real:

"The measured synthetic benchmark reduction is 96.43 percent in individual model evaluations. I would describe 20 percent as a conservative target or resume-level summary, and I would always specify the benchmark setting. I would not claim that exact reduction across all models until broader experiments are run."

If asked how this relates to adversarial vulnerability:

"The current contribution is not a full adversarial attack study. It implements explanation drift and sensitivity metrics. The idea is that if small perturbations preserve the label but significantly change the SIS mask, that suggests brittle evidence use and motivates adversarial evaluation."

## Addendum: new measured digits MLP result

The repository now includes and has run `experiments/run_nn_sis_experiments.py` on sklearn digits with `--model mlp --max-examples 1 --seed 0`.

Measured neural-network result:

- Train accuracy: 0.9911
- Test accuracy: 0.9733
- Selected example: test index 187, target class 4
- Original confidence: 0.9999985
- Threshold: 0.8
- Original SIS evaluations: 7945
- SHAP-guided SIS evaluations: 199
- SHAP-guided evaluation reduction: 97.50 percent
- SHAP-guided final confidence: 0.998905

Updated safe wording:

"Implemented an SIS-based neural-network explainability framework extending Google Research's SIS code with SHAP-inspired acceleration, probabilistic sampling, hierarchical masks, and stability metrics; validated the framework on a sklearn digits MLP and measured a 97.50 percent reduction in individual model evaluations for SHAP-guided SIS on one high-confidence example while preserving threshold sufficiency."

Still avoid claiming MNIST/CIFAR/CNN/adversarial results until those scripts are run and saved.
