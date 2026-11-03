# Model comparison (test)

Mean ± standard deviation over the configured seeds. F1 is macro F1 on the fixed test split.
Inference time is milliseconds per image; parameter count is in millions; training time is minutes per run.

| Model | Seeds | Macro F1 | Accuracy | Params (M) | Inference (ms/image) | Train (min) |
|---|---|---:|---:|---:|---:|---:|
| resnet50 | 42,43,44 | 0.925 ± 0.011 | 0.925 ± 0.010 | 23.53 ± 0.00 | 1.87 ± 0.33 | 2.54 ± 0.03 |
| vit_b16 | 42,43,44 | 0.852 ± 0.017 | 0.852 ± 0.015 | 85.81 ± 0.00 | 5.97 ± 0.23 | 13.44 ± 1.78 |
| convnext_t | 42,43,44 | 0.924 ± 0.010 | 0.925 ± 0.010 | 27.83 ± 0.00 | 2.13 ± 0.01 | 4.61 ± 0.29 |
