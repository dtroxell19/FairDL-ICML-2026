########################################################################################################################
#This script defines hyperparameters and the evaluation criteria/function used for the loan experiments
########################################################################################################################

def get_slack():
    """
    Defines how strict the constraints are for loan experiments
    """
    return 0.01


def get_num_batches_train():
    """
    Defines number of minibatches when training
    """
    return 1


def get_num_batches_test():
    """
    Defines number of minibatches when training
    """
    return 1
