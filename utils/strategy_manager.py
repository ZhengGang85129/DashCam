#!/usr/bin/env python3
import yaml
from typing import Dict

SCHEDULER = ['CosineAnnealingLR', 'ReduceLRONPlateau']
OPTIMIZER = ['adamw', 'radam']
class Strategy_Mananger:
    
    def __init__(self, confDICT: Dict):
        
        self.batch_size = int(confDICT.get('batch_size', 10))
       
        self.resume = bool(confDICT.get('resume', False))
        self.optim_resume = bool(confDICT.get('optim_resume', False))
       
        self.trainer_name = str(confDICT.get('trainer_name', 'trainer'))
       
        self.optimizer = confDICT.get('optimizer', None)
        self.scheduler = confDICT.get('scheduler', None)
        self.model_dir = confDICT.get('model_dir', 'model_dir')
        self.monitor_dir = confDICT.get('monitor_dir', 'monitor_dir')
        self.trainable_parts = confDICT.get('trainable_parts', ["*"])
        self.check_point_path = confDICT.get('check_point_path', None)
        self.optim_check_point_path = confDICT.get('optim_check_point_path', None)
        self.gamma = confDICT.get('gamma', 0)
        self.gamma_max = confDICT.get('gamma_max', 0)
        self.decay_coefficient = confDICT.get('decay_coefficient', 30) 
        self.unfreezing = confDICT.get('unfreezing', None) 
        self.early_stopping = confDICT.get('early_stopping', None) 
        #if self.scheduler["name"] not in SCHEDULER: raise ValueError(f"No such scheduler yet: {self.scheduler['name']}") 
        self.stride = confDICT.get('stride', 2) 
        self.reweight = confDICT.get('reweight')
        self.evaluation_check_point_path = confDICT.get('evaluation_check_point_path', None)
        if self.optimizer["name"] not in OPTIMIZER: raise ValueError(f"No such scheduler yet: {self.optimizer['name']}") 

def yaml_content_args(yamlCONTENT) -> Strategy_Mananger:
    return Strategy_Mananger(yaml.safe_load(yamlCONTENT))   

def get_strategy_manager(yamlFILE: str):
    print(yamlFILE)
    with open(yamlFILE, 'r') as FIN:
        return yaml_content_args(FIN) 
    
    
if __name__ == "__main__":
    
    strategy_manager = get_strategy_manager("configs/training-single.yaml")                     
    print(strategy_manager.optimizer)
    print(strategy_manager.scheduler)
    print(strategy_manager.check_point_path) 
    print(strategy_manager.trainable_parts) 
