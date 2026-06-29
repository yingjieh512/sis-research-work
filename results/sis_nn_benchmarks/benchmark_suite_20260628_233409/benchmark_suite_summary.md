# SIS Neural-Network Benchmark Suite

This report is generated from local benchmark runs. A dataset is marked `unavailable` or `failed` when dependencies, downloads, or runtime conditions prevent the benchmark from completing.

| Dataset | Status | Notes |
| --- | --- | --- |
| digits | completed | accuracy 0.9733; output `C:\Users\Yingjie Huang\Downloads\sis-research-work\results\sis_nn_benchmarks\benchmark_suite_20260628_233409\digits_mlp` |
| mnist | completed | accuracy 0.8940; output `C:\Users\Yingjie Huang\Downloads\sis-research-work\results\sis_nn_benchmarks\benchmark_suite_20260628_233409\mnist_mlp` |

## digits

| Method | Threshold rate | Mean confidence | Mean subset | Mean evals | Eval reduction | Mean runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hierarchical_sis | 1.000 | 0.963973 | 2.00 | 40.0 | 99.50% | 0.0043 |
| original_sis | 1.000 | 0.999998 | 4.00 | 7945.0 | 0.00% | 0.0506 |
| probabilistic_sis | 1.000 | 0.998905 | 3.00 | 600.0 | 92.45% | 0.0113 |
| shap_guided_sis | 1.000 | 0.998905 | 3.00 | 199.0 | 97.50% | 0.0036 |

## mnist

| Method | Threshold rate | Mean confidence | Mean subset | Mean evals | Eval reduction | Mean runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hierarchical_sis | 1.000 | 0.821547 | 2.00 | 121.0 | 99.79% | 0.0093 |
| original_sis | 1.000 | 0.998483 | 10.00 | 56370.0 | 0.00% | 0.2341 |
| probabilistic_sis | 1.000 | 0.994795 | 6.00 | 1218.0 | 97.84% | 0.0280 |
| shap_guided_sis | 1.000 | 0.994795 | 6.00 | 405.0 | 99.28% | 0.0098 |

## Honesty Note

Only rows with status `completed` are measured benchmark evidence. MNIST should be discussed as measured only when its row is completed and the referenced per-dataset result files are present.
