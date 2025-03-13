from models.model import DSA_RNN
from models.model import baseline_model
from models.model import timesformer
import torch.nn as nn
def get_model(model_type:str = 'baseline_model')->nn.Module:
    '''
    Factory function to create different types of models
    '''
    
    if model_type.lower() == 'baseline_model':
        return baseline_model()
    elif model_type.lower() == 'das_rnn':
        return DSA_RNN()
    elif model_type.lower() == 'timesformer':
        return timesformer()
    else:
        raise RuntimeError(f'No such model type: {model_type.lower()}')