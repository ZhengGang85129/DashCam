from typing import Dict, List
import torch.nn as nn
import torch.optim.optimizer
import torch.optim.optimizer
import torch.optim.sgd
import torch

def split_params(module):
    """
    Divide module into two different parts:
    - normal params: (with weight decay)
    - no_decay params: (can't use weight decay)
    """
    
    decay = []
    no_decay = []
    
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
            for param in m.parameters(recurse = False):
                no_decay.append(param)
        else:
            for param in m.parameters(recurse = False):
                decay.append(param)
    return decay, no_decay

def print_optimizer_param_groups(optimizer):
    print("Optimizer Parameter Groups Summary:")
    for idx, group in enumerate(optimizer.param_groups):
        lr = group['lr']
        print(f"Group {idx}: lr = {lr}")
    return 

def get_optimizer(model: nn.Module, optimizer: Dict)->torch.optim.Optimizer:
    
    if optimizer['name'].lower() == 'radam':
        '''
        Should check the logic.
        '''
        param_groups = []
        for layer_name, base_lr in optimizer['differential_lr'].items():
            layer = getattr(model.backbone, layer_name)
            lr = base_lr
            if isinstance(lr, str):
                lr = float(lr)
            param_groups.append({'params': [p for _, p in layer if p.requires_grad], 'lr': lr})
        
        return torch.optim.RAdam(params = param_groups, lr = float(optimizer['lr']))
    
    elif optimizer['name'].lower() == 'adamw':
        param_groups = []
        weight_decay = optimizer['weight_decay']
        if isinstance(weight_decay, str):
            weight_decay = float(weight_decay)
            
        for layer_name, base_lr in optimizer['differential_lr'].items():
            if isinstance(base_lr, str):
                base_lr = float(base_lr)
            layer = getattr(model.backbone, layer_name)
            decayed_p, no_decayed_p = split_params(layer)
            if decayed_p:
                param_groups.append({
                    'params': decayed_p,
                    'lr': base_lr,
                })
            if no_decayed_p:
                param_groups.append({
                    'params': no_decayed_p,
                    'lr': base_lr,
                })    
        
    
        return torch.optim.AdamW(param_groups, weight_decay = weight_decay)
    
    raise ValueError(f'No such type of optimizer in this repo: {type}.')