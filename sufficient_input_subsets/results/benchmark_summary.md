# SIS Extension Benchmark Summary

Synthetic image benchmark with a deterministic confidence function.

| Method | Runtime (s) | f_batch calls | Individual evals | Subset size | Score | Met threshold | Speedup vs baseline | Stability |
| --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: |
| original_sis | 0.0051 | 128 | 3974 | 7 | 0.9997 | yes | 0.00% | 0.833 |
| shap_guided_sis | 0.0016 | 19 | 142 | 7 | 0.9997 | yes | 96.43% | 0.750 |
| probabilistic_sis | 0.0081 | 100 | 715 | 8 | 0.9999 | yes | 82.01% | 0.833 |
| hierarchical_sis | 0.0028 | 46 | 129 | 7 | 0.9997 | yes | 96.75% | 0.750 |

## SHAP-guided overhead reduction

Measured reduction in individual model evaluations: **96.43%**.

SHAP-guided SIS achieved at least a 20% individual-evaluation reduction on this benchmark.

Next optimization hooks if the measured reduction is below target: reduce `max_candidates`, use larger image-region feature groups before pixel refinement, reuse importance caches across nearby perturbations, and batch larger candidate sets when the model supports it.
