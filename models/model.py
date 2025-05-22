from .baseline_model import baseline_model
from .timesformer import timesformer
from .xai_accident import AccidentXai
from .swintransformer import swintransformer
from .model_cnn_lstm import CNN_LSTM
from .mvit_v2 import mvit_v2
def get_model(model_type: str):
    
    if model_type.lower() == 'baseline':
        return baseline_model
    elif model_type.lower() == 'timesformer':
        return timesformer
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
    
    
