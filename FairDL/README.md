# FairDL

This repository contains the code for numerical experiments described in the paper "Differentiable Optimization Layers for Guaranteed Fairness in Deep Learning." Each folder in the repository corresponds to one of numerical experiments, and each operates independently (i.e. self-contained and has its own README.md).

# Getting Started
**Note:** This repository uses Git LFS for large model files. Make sure Git LFS is installed before cloning:

1. Install Git LFS (if not already installed):
```bash
git lfs install
```

2. Clone the repo with LFS objects
```bash
git clone https://github.com/dtroxell19/FairDL.git
cd FairDL
git lfs pull
```

3. To install the required Python packages, please use **Python 3.12.9** and follow these steps:

(Conda)
- 3.1 Create a new environment with Python 3.12.9
```bash
conda create -n FairDL_env python=3.12.9 -y
```
- 3.2 Activate Environment
```bash
conda activate FairDL_env
```
- 3.3 Upgrade pip and install packages
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Alternatively, if you prefer venv, after having Python 3.12 installed, do the following:

(venv)
- 3.1  Create the virtual environment with Python 3.12.9
```bash
python3.12 -m venv FairDL_env
```

- 3.2 Activate (Windows)
```bash
source FairDL_env/bin/activate
```

- 3.2 Activate (Mac)
```bash
FairDL_env\Scripts\activate 
```
- 3.3 Upgrade pip and install packages
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
