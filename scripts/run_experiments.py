from scripts.train import train_fn, get_logger, get_device
import optuna
from utils.YamlArguments import load_yaml_file_from_arg
import sys
from utils.strategy_manager import get_strategy_manager
from utils.scheduler import get_scheduler

def objective(trial):
    
    args = load_yaml_file_from_arg('./experiment/mvit2.yaml')
    
    logger = get_logger()
    device = get_device()
    
    if args is None:
        raise ValueError('Please provide configuration file.')
     
    if args.augmentation_types:
        print(f"Using augmentation types: {args.augmentation_types}")
        print(f"Global augmentation probability: {args.augmentation_prob}")
    
    manager = get_strategy_manager(args.training_strategy)
    
    manager.optimizer['differential_lr']['classifier'] = trial.suggest_float("classifier_lr", 1e-5, 1e-2, log = True)
    
    manager.optimizer['weight_decay'] = trial.suggest_float('weight_decay', 1e-4, 1e-2, log = True) 
    
    manager.batch_size = trial.suggest_categorical("batch_size", [4, 8, 12])
    
    manager.stride = trial.suggest_categorical("stride", [1, 2, 3, 4])
    
    depth = trial.suggest_int("depth",1, 3)
    dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.3, step = 0.05)
    hidden_dim = trial.suggest_int("hidden_dim", 64, 512, step = 64)
    args.classifier = [] 
    args.hidden_dim = hidden_dim
    args.dropout_rate = dropout_rate
    args.depth = depth
     
    for L in range(depth):
        args.classifier.append(hidden_dim)
        args.classifier.append('relu')
        if L != depth -1:
            args.classifier.append(f'dropout:{dropout_rate}')
    return train_fn(args = args, manager = manager, logger = logger, device = device) 

def run_optuna():
    study = optuna.create_study(direction = 'minimize')
    study.optimize(objective, n_trials = 50)

    print("Best parameters:", study.best_params)

if __name__ == "__main__":
    run_optuna()