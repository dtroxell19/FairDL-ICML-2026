import torch.nn as nn

def get_ffnn_structure():
    '''
    Feed-forward NN structure used for all models in synthetic experiments
    '''
    # overparameterized NN for synthetic experiments (about 3M parameters)
    return nn.Sequential(
        nn.Linear(151, 800),
        nn.LayerNorm(800),
        nn.ReLU(),

        nn.Linear(800, 600),
        nn.LayerNorm(600),
        nn.ReLU(),

        nn.Linear(600, 600),
        nn.LayerNorm(600),
        nn.ReLU(),

        nn.Linear(600, 600),
        nn.LayerNorm(600),
        nn.ReLU(),

        nn.Linear(600, 600),
        nn.LayerNorm(600),
        nn.ReLU(),

        nn.Linear(600, 600),
        nn.LayerNorm(600),
        nn.ReLU(),

        nn.Linear(600, 600),
        nn.LayerNorm(600),
        nn.ReLU(),

        nn.Linear(600, 600),
        nn.LayerNorm(600),
        nn.ReLU(),

        nn.Linear(600, 400),
        nn.LayerNorm(400),
        nn.ReLU(),

        nn.Linear(400, 300),
        nn.LayerNorm(300),
        nn.ReLU(),

        nn.Linear(300, 200),
        nn.LayerNorm(200),
        nn.ReLU(),

        nn.Linear(200, 100),
        nn.LayerNorm(100),
        nn.ReLU(),

        nn.Linear(100, 50),
        nn.LayerNorm(50),
        nn.ReLU(),
        
        nn.Linear(50, 10),
        nn.LayerNorm(10),
        nn.ReLU(),

        nn.Linear(10, 3),
        nn.LayerNorm(3),
        nn.ReLU(),

        nn.Linear(3, 1)
    )