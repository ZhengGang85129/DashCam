
import torch.optim.optimizer
import torch.optim.optimizer
import torch.optim.sgd


def get_optimizer(type = str)->torch.optim.Optimizer:
    
    if type.lower() == 'radam':
        return torch.optim.RAdam
    elif type.lower() == 'adam':
        return torch.optim.Adam
    elif type.lower() == 'sgd':
        return torch.optim.sgd
    
    raise ValueError(f'No such type of optimizer in this repo: {type}.')