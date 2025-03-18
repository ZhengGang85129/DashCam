from .dsa_rnn import DSA_RNN
from .baseline_model import baseline_model
from .timesformer import timesformer
from .xai_accident import AccidentXai
from .swintransformer import swintransformer
def get_model(model_type: str):
    
    if model_type.lower() == 'baseline':
        return baseline_model
    elif model_type.lower() == 'timesformer':
        return timesformer
    elif model_type.lower() == 'accidentxai':
        return AccidentXai 
    elif model_type.lower() == 'swintransformer':
        return swintransformer
    else:
        raise ValueError(f'No such model yet: {model_type}')
    
    