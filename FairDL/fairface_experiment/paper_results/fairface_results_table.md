**FairFace test set results across architectures and fairness methods.** ΔAcc shows percent change in accuracy relative to the F-Layer (ε = 0.0001).

| Backbone | Method | Accuracy | ΔAcc (%) |
|:---------|:-------|:--------:|:--------:|
| ResNet-18 | F-Layer | 0.7616 | — |
|  | Projection | 0.7479 | -1.81 |
|  | Penalty | 0.5496 | -27.84 |
|  | Strict Penalty | 0.5496 | -27.84 |
| SimpleCNN | F-Layer | 0.6729 | — |
|  | Projection | 0.6703 | -0.39 |
|  | Penalty | 0.5496 | -18.33 |
|  | Strict Penalty | 0.5496 | -18.33 |
| ViT-B/16 (LoRA) | F-Layer | 0.7749 | — |
|  | Projection | 0.7541 | -2.69 |
|  | Penalty | 0.5496 | -29.08 |
|  | Strict Penalty | 0.5496 | -29.08 |
| DenseNet-121 | F-Layer | 0.7623 | — |
|  | Projection | 0.7389 | -3.07 |
|  | Penalty | 0.5506 | -27.77 |
|  | Strict Penalty | 0.5506 | -27.77 |
| Swin-T | F-Layer | 0.8081 | — |
|  | Projection | 0.7646 | -5.38 |
|  | Penalty | 0.5506 | -31.86 |
|  | Strict Penalty | 0.5506 | -31.86 |