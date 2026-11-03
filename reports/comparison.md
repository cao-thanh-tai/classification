# Model comparison (validation)

Mean ± standard deviation over the configured seeds. F1 is macro F1 on the fixed validation split.
Inference time is milliseconds per image; parameter count is in millions; training time is minutes per run.

| Model | Seeds | Macro F1 | Accuracy | Params (M) | Inference (ms/image) | Train (min) |
|---|---|---:|---:|---:|---:|---:|
| resnet18 | 42,43,44 | 0.928 ± 0.006 | 0.927 ± 0.005 | 11.18 ± 0.00 | 0.67 ± 0.01 | 1.49 ± 0.04 |
| resnet50 | 42,43,44 | 0.953 ± 0.007 | 0.953 ± 0.007 | 23.53 ± 0.00 | 1.70 ± 0.02 | 2.54 ± 0.03 |
| vit_b16 | 42,43,44 | 0.875 ± 0.024 | 0.874 ± 0.022 | 85.81 ± 0.00 | 21.14 ± 3.82 | 13.44 ± 1.78 |
| vit_b32 | 42,43,44 | 0.595 ± 0.007 | 0.598 ± 0.009 | 87.46 ± 0.00 | 1.65 ± 0.24 | 2.95 ± 0.31 |
| convnext_t | 42,43,44 | 0.959 ± 0.002 | 0.959 ± 0.002 | 27.83 ± 0.00 | 2.42 ± 0.25 | 4.61 ± 0.29 |
