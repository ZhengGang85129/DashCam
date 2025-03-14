from torch import optim

def get_optimizer(name: str)->optim.Optimizer:
    if name.lower() == 'sgd':
        return optim.SGD
    elif name.lower() == 'adam':
        return optim.Adam
    elif name.lower() == 'adamw':
        return optim.AdamW
    else:
        return optim.RAdam #Default
    