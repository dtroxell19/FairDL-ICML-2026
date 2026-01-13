# Description
This folder contains datasets, trained models, and code relating to the Synthetic Experiments section in the Paper. 

- fair_model.py, baseline_model.py, penalty_model.py: Python scripts that contain class definitions for the F-Layer, Projeciton, and Penalty/Strict Penalty models,respectively
- data_generation.py: Script that generates the 32 synthetic datasets listed in the datasets/ folder
- synthetic_experiments.py: Main driver script that trains models of specified type for all 32 synthetic datasets
- plotting.ipynb: Notebook that replicates plots used in the paper
- fairness_penalty_synthetic_experiments.py: script that returns the hyperparameters used for each dataset for the Penalty model after cross validation
- configs/nn_architecture and configs/synthetic_xp_params.py: Configurations on model architecture and epsilon/batch sizes/etc. used in the synthetic dataset experiments
- trained_models_small_training_batch/ and trained_models_large_training_batch/ : Folders containing all 32 models for each of the 4 model types when training batch size was 20 and 2000, respectively

# Re-creating Results
**Note:** This repository uses Git LFS for large model files. Make sure Git LFS is installed before cloning:

1. Complete the instructions under "Getting Started" in the parent directory's README.md
2. Run the following with {model_type} as fair, baseline, penalty, or strict penalty:

```bash
python synthetic_experiments.py --model_type {model_type}
```

3. Alternatively, run the cells in plotting.ipynb to re-create plots shown in the paper. By default, this notebook uses the .csv result files already in the GitHub repository

4. (Optional) To run the code with your own configurations (if desired), change the values in configs/nn_architecture.py and synthetic_xp_params.py

