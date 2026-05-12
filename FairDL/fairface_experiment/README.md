# Description
This folder contains datasets and code relating to the FairFace Experiments.

- load_data.py: Script that loads the FairFace dataset from HuggingFace and prepares train/val/test splits
- configs/: Configuration files for model architecture and experiment parameters

# Dataset
FairFace (Karkkainen & Joo, WACV 2021) is a face attribute dataset containing ~108k images balanced across 7 race groups, 2 gender classes, and 9 age groups. We load it from HuggingFace (`HuggingFaceM4/FairFace`).

## Experiment Setup
- **Prediction target:** Binary age classification (0 = under 30, 1 = 30 or older)
  - The original 9-class ages are binarized at the 30-year boundary: classes "0-2", "3-9", "10-19", "20-29" map to 0; classes "30-39" through "more than 70" map to 1
- **Protected attribute:** Gender (0 = Male, 1 = Female)
- **Image preprocessing:** Resized to 224x224, ImageNet-normalized (for ResNet backbone)

# Re-creating Results

1. Complete the instructions under "Getting Started" in the parent directory's README.md
2. Load and preprocess the data:

```bash
python load_data.py
```
This saves preprocessed PyTorch TensorDatasets to a `fairface_splits/` folder

3. Perform cross-validation, train models, and aggregate results via:

```bash
./run_all.sh
```