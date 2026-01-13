def get_slack():
    '''
    Defines how different the average prediction can be for 2 groups for a given minibatch 
    '''
    return .05

def get_val_size():
    '''
    Defines how large the validation set is
    '''
    return 10000

def get_num_batches_train():
    '''
    Defines number of minibatches when training
    '''
    return 1200

def get_num_batches_test():
    '''
    Defines number of minibatches when training
    '''
    return 3

def b_tau():
    '''
    Minimum threshold for batch size
    '''
    return 1000

def epsilon_0():
    '''
    Minimum threshold for batch size
    '''
    return .1