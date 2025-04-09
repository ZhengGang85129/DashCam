import torch
from typing import Dict, Union


def get_scheduler(scheduler: Dict[str, Union[str, int]], optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:

    if scheduler['name'] == 'CosineAnnealingLR':
        
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer = optimizer,
            T_max = float(scheduler['T_max']),
            eta_min = float(scheduler['eta_min']),
            verbose = True
        )
    
    elif scheduler['name'] == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer = optimizer,
            mode = scheduler['mode'],
            factor = float(scheduler['factor']),
            patience = int(scheduler['patience']),
            min_lr = float(scheduler['min_lr']) ,
            verbose = True
        )
        
    else:
        raise ValueError('Opimizer')   
        