# Description
This folder contains datasets and code relating to the Employee Performance Experiments section in the paper. 

- fair_model_emp.py, baseline_model_emp.py: Python scripts that contain class definitions for the F-Layer and Projection models,respectively. The Penalty model is accounted for via the augmented loss function defined in emp_experiments.py
- employee_performance_data.py: Script that generates the X.pkl and y.pkl used to train the models
- emp_experiments.py: Main driver script that trains models of specified type
- configs/emp_nn_architecture.py and configs/emp_xp_params.py: Configurations on model architecture and epsilon/batch sizes/etc. used in the employee performance experiments
- Employee_Performance.xls: raw dataset used in the experiments

# Re-creating Results

1. Complete the instructions under "Getting Started" in the parent directory's README.md
2. Run the following with {model_type} as fair, baseline, or penalty:

```bash
python emp_experiments.py --model_type {model_type}
```
3. The results will be saved to fairness_results.csv
4. (Optional) To run the code with your own configurations, change the values in configs/emp_nn_architecture.py and emp_xp_params.py