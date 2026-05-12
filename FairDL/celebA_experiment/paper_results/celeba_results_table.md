**CelebA (Smiling) test set results across architectures and fairness methods.** ΔAcc shows percent change in accuracy relative to the F-Layer (ε = 0.0001).

| Backbone | Method | Accuracy | ΔAcc (%) |
|:---------|:-------|:--------:|:--------:|
| ResNet-18 | F-Layer | 0.9038 | — |
|  | Projection | 0.8852 | -2.10 |
|  | Penalty | 0.4997 | -80.85 |
|  | Strict Penalty | 0.5014 | -80.25 |
| SimpleCNN | F-Layer | 0.8141 | — |
|  | Projection | 0.8088 | -0.65 |
|  | Penalty | 0.5014 | -62.35 |
|  | Strict Penalty | 0.5014 | -62.35 |
| ViT-B/16 (LoRA) | F-Layer | 0.9062 | — |
|  | Projection | 0.8760 | -3.45 |
|  | Penalty | 0.5014 | -80.74 |
|  | Strict Penalty | 0.5014 | -80.73 |
| DenseNet-121 | F-Layer | 0.9149 | — |
|  | Projection | 0.8930 | -2.46 |
|  | Penalty | 0.5014 | -82.47 |
|  | Strict Penalty | 0.5014 | -82.47 |
| Swin-T | F-Layer | 0.9177 | — |
|  | Projection | 0.8882 | -3.32 |
|  | Penalty | 0.5014 | -83.02 |
|  | Strict Penalty | 0.5014 | -83.02 |