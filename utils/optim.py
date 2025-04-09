from typing import Dict, List
import torch.nn as nn
import torch.optim.optimizer
import torch.optim.optimizer
import torch.optim.sgd
import torch

def get_optimizer(optimizer: Dict, params: List[nn.Parameter])->torch.optim.Optimizer:
    
    if optimizer['name'].lower() == 'radam':
        params_to_be_updated = [p for _, p in params if p.requires_grad]
        
        return torch.optim.RAdam(params = params_to_be_updated, lr = float(optimizer['lr']))
    elif optimizer['name'].lower() == 'adamw':
        decay = []
        no_decay = []
        for name, param in params:
            if not param.requires_grad:
                continue
            if any(nd in name for nd in ["bias", "bn", "norm"]):
                no_decay.append(param)
            else:
                decay.append(param)
        params_to_be_updated = [
            {'params': decay, 'weight_decay': float(optimizer['weight_decay'])},
            {'params': no_decay, 'weight_decay': 0.0}
        ]
    
        
        return torch.optim.AdamW(params = params_to_be_updated, lr = float(optimizer['lr']))
    
    raise ValueError(f'No such type of optimizer in this repo: {type}.')