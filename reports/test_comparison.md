# Model comparison (test)

Mean ± standard deviation over the configured seeds. F1 is macro F1 on the fixed test split.
Inference time is milliseconds per image; parameter count is in millions; training time is minutes per run.

| Model | Seeds | Macro F1 | Accuracy | Params (M) | Inference (ms/image) | Train (min) |
|---|---|---:|---:|---:|---:|---:|
| resnet18 | 42,43,44 | 0.882 ± 0.014 | 0.883 ± 0.014 | 11.18 ± 0.00 | 0.74 ± 0.08 | 1.49 ± 0.04 |
| resnet50 | 42,43,44 | 0.925 ± 0.011 | 0.925 ± 0.010 | 23.53 ± 0.00 | 1.78 ± 0.12 | 2.54 ± 0.03 |
| vit_b16 | 42,43,44 | 0.852 ± 0.017 | 0.852 ± 0.015 | 85.81 ± 0.00 | 6.11 ± 0.20 | 13.44 ± 1.78 |
| vit_b32 | 42,43,44 | 0.544 ± 0.026 | 0.553 ± 0.024 | 87.46 ± 0.00 | 1.30 ± 0.00 | 2.95 ± 0.31 |
| convnext_t | 42,43,44 | 0.924 ± 0.010 | 0.925 ± 0.010 | 27.83 ± 0.00 | 2.15 ± 0.05 | 4.61 ± 0.29 |
