from .baseline_model import baseline_model
#from .timesformer import timesformer
from .xai_accident import AccidentXai
from .swintransformer import swintransformer
from .model_cnn_lstm import CNN_LSTM
from .mvit_v2 import mvit_v2
import torch.nn as nn
import torch
import os
def get_model(model_type: str):
    
    if model_type.lower() == 'baseline':
        return baseline_model
    #elif model_type.lower() == 'timesformer':
    #    return timesformer
    elif model_type.lower() == 'accidentxai':
        return AccidentXai 
    elif model_type.lower() == 'swintransformer':
        return swintransformer
    elif model_type.lower() == 'cnn_lstm':
        return CNN_LSTM
    elif model_type.lower() == 'mvit_v2':
        return mvit_v2 
    else:
        raise ValueError(f'No such model yet: {model_type}')
    
def load_model(args, manager) -> nn.Module:
    model = get_model(args.model_type)(classifier = args.classifier)
    if manager.evaluation_check_point_path is None:
        raise FileNotFoundError(f"The state_dict is None.")
    elif not os.path.isfile(manager.evaluation_check_point_path):
        raise FileNotFoundError(f"The state_dict {manager.evaluation_check_point_path} doesn't exist.")
    saved_state = torch.load(manager.evaluation_check_point_path, map_location = 'cpu')

    if isinstance(saved_state, dict):
        if 'model_state_dict' in saved_state:
            model.load_state_dict(saved_state['model_state_dict'])
        else:
            model.load_state_dict(saved_state)
    else:
        model = saved_state
    model.to(args.device) 
    return model
    
