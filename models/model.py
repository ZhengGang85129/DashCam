from .dsa_rnn import DSA_RNN
from .baseline_model import baseline_model
from .timesformer import timesformer

def get_model(model_type: str):
    
    if model_type == 'baseline':
        return baseline_model
    elif model_type == 'timesformer':
        return timesformer
    else:
        raise ValueError(f'No such model yet: {model_type}')
    
    