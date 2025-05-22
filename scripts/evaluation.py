import torch
import torch.nn as nn 
from utils.tool import get_device
from typing import Optional
import torch.nn.functional as F
import os
from tqdm import tqdm
from models.model import get_model
import matplotlib.pyplot as plt
from src.datasets.accident_dataset import AccidentDataset
from utils.YamlArguments import load_yaml_file_from_arg
import sys
from utils.strategy_manager import get_strategy_manager
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
from pathlib import Path

def load_model(args, manager) -> nn.Module:
    model = get_model(args.model_type)(classifier = args.classifier)
    if manager.evaluation_check_point_path is None:
        raise FileNotFoundError(f"The state_dict is None.")
    elif not os.path.isfile(manager.evaluation_check_point_path):
        raise FileNotFoundError(f"The state_dict {manager.evaluation_check_point_path} doesn't exist.")
    saved_state = torch.load(manager.evaluation_check_point_path, map_location = args.device if args.device is not None else 'cpu')

    if isinstance(saved_state, dict):
        if 'model_state_dict' in saved_state:
            model.load_state_dict(saved_state['model_state_dict'])
        else:
            model.load_state_dict(saved_state)
    else:
        model = saved_state
    
    return model

#def get_dataloader()-> torch.utils.data.DataLoader:
def get_dataloader(args, manager):

    dataset = AccidentDataset(
        root_dir = args.test_dir,
        csv_file = args.test_csv,
        mode = 'evaluation',
        model_type = args.model_type,
        stride = 1,
        frame_per_window = 16,
        
    )
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size = 16,
        num_workers = manager.batch_size,
        pin_memory = True
    )
    return loader

def plot_prob(probs, manager):

    plt.hist(probs)
    plt.savefig(Path(manager.monitor_dir) / f'likelihood_{manager.trainer_name}.pdf')
    plt.savefig(Path(manager.monitor_dir) / f'likelihood_{manager.trainer_name}.png')
 
def evaluate_fn(args, manager) -> None:
    
    loader = get_dataloader(args, manager=manager) 
    print(len(loader))  
    model = load_model(args, manager) 
    model.to(args.device)
    
    probs = []
    ids = []
    for batch in tqdm(loader):
        frames, frame_ids = batch
        frames = frames.to(args.device)
        
        with autocast():
            output = model(frames)
            prob = F.softmax(output, dim = 1)
            probs.append(prob)
            ids.append(frame_ids)
    probs = torch.cat(probs).flatten()
    ids = torch.cat(ids).flatten()
    dataframe = []
    
    probs_list = [] 
    for id, prob in zip(ids, probs):
        dataframe.append({
            "id": id.item(),
            "score": prob.item()
        })
        probs_list.append(prob.detach().to('cpu')) 
    submission = pd.DataFrame(dataframe)    
    submission.to_csv(f'submission-{manager.trainer_name}.csv' if manager.trainer_name else 'submission.csv', index = False)
    plot_prob(probs= probs_list, manager = manager)
    
def main():
    
    use_yaml_file = len(sys.argv) == 1+1 and '.yaml' in sys.argv[1]
    args = load_yaml_file_from_arg(sys.argv[1]) if use_yaml_file else None
    if args is None:
        raise ValueError('Please provide configuration file.')
    
    manager = get_strategy_manager(args.training_strategy)
    args.device = get_device() 
    evaluate_fn(args = args, manager=manager) 
    
        
if __name__  == '__main__':
    
    main()