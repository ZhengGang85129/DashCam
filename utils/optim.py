
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
    elif type.lower() == 'nadam':
        return torch.optim.NAdam
    elif type.lower() == 'adamw':
        return torch.optim.AdamW
    elif type.lower() == 'lion':
        from lion_pytorch import Lion
        return Lion
    elif type.lower() == 'ranger21':
        from ranger21 import Ranger21
        return Ranger21

    raise ValueError(f'No such type of optimizer in this repo: {type}.')
