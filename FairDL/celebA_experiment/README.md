# CelebA Experiment

This folder contains datasets and code for the CelebA fairness experiments, mirroring the
structure used in the FairFace experiments.

## Dataset

**CelebA** (Liu et al., ICCV 2015) is a large-scale face attribute dataset containing
~202k celebrity images, each annotated with 40 binary attributes.
We load it from HuggingFace (`flwrlabs/celeba`).

## Experiment Setup

- **Prediction target:** `Smiling` (binary attribute)
- **Protected attributes:** `Male` × `Young` → 4 intersectional groups
- **Fairness constraint:** Pairwise demographic parity across all C(4,2) = 6 group pairs
- **Image preprocessing:** Resized to 224×224, ImageNet-normalized (for pretrained backbones)

### Intersectional Groups

| Group ID | Gender   | Age       |
|----------|----------|-----------|
| 0        | Female   | Not Young |
| 1        | Female   | Young     |
| 2        | Male     | Not Young |
| 3        | Male     | Young     |

## Re-creating Results

1. Complete the instructions under "Getting Started" in the parent directory's `README.md`
2. Load and preprocess the data:

```bash
python load_data.py
```

3. This saves preprocessed PyTorch TensorDatasets to a `celeba_splits/` folder
4. Run all experiments:

```bash
chmod +x run_all.sh
./run_all.sh
```