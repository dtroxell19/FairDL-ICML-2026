# Description
This folder contains datasets and code relating to the Loan Experiments section in the paper. 

- fair_model_loan.py, baseline_model_loan.py: Python scripts that contain class definitions for the F-Layer and Projection models, respectively. The Strict Penalty model is accounted for via the augmented loss function defined in run_loan_experiments.py
- load_data.py: Script that loads datasets from Kaggle used for the loan experiments
- cross_val.py: Script that helps pick lambda hyperparameter weighting term for Penalty model. Results showed high sensitivity, so only Strict Penalty model used (as discussed in paper)
- run_loan_experiments.py: Main driver script that trains models of specified type
- configs/loan_nn_architecture.py and configs/loan_xp_params.py: Configurations on model architecture and epsilon/batch sizes/etc. used in the loan  experiments
- plotting.ipynb: Jupyter notebook used to generate plots shown in paper
- loan_models/ directory stores the trained models for all 3 methods

# Re-creating Results

1. Complete the instructions under "Getting Started" in the parent directory's README.md
2. Run the following with {model_type} as fair, baseline, or strict_penalty:

```bash
python run_loan_experiments.py --model_type {model_type}
```
3. The results will be saved to a results/ folder
4. (Optional) To run the code with your own configurations, change the values in configs/loan_nn_architecture.py