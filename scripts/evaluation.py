import torch
import torch.nn as nn 
from utils.tool import get_device
import torch.nn.functional as F
import os
from tqdm import tqdm
from models.model import get_model, load_model
import matplotlib.pyplot as plt
from src.datasets.accident_dataset import AccidentDataset
from utils.YamlArguments import load_yaml_file_from_arg
import sys
from utils.strategy_manager import get_strategy_manager
from torch.cuda.amp import autocast
import pandas as pd
from pathlib import Path
import numpy as np

#def get_dataloader()-> torch.utils.data.DataLoader:
def get_dataloader(args, manager):

    dataset = AccidentDataset(
        root_dir = args.test_dir,
        csv_file = args.test_csv,
        mode = 'evaluation',
        model_type = args.model_type,
        stride = manager.stride,
        frame_per_window = 16,
        
    )
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size = 8,
        num_workers = args.num_workers ,
        pin_memory = False
    )
    return loader

def plot_prob(probs, manager):

    plt.hist(probs)
    plt.savefig(Path(manager.monitor_dir) / f'likelihood_{manager.trainer_name}.pdf')
    plt.savefig(Path(manager.monitor_dir) / f'likelihood_{manager.trainer_name}.png')
 
def evaluate_fn(args, manager) -> None:
    manager.evaluation_check_point_path = 'mlartifacts/0/3e09264b4c904dd499be80b37e3a05ab/artifacts/model/mvit_v2mvit2_weighted_positive/data/model.pth' 
    loader = get_dataloader(args, manager=manager) 
    print(len(loader))  
    model = load_model(args, manager) 
    print(torch.cuda.memory_allocated(args.device) / (1024 ** 2) ) 
    probs = []
    ids = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader):
            frames, frame_ids = batch
            frames = frames.to(args.device)
            
            with autocast():
                output = model(frames)
                prob = F.softmax(output, dim = 1)
                probs.append(prob[..., 1].cpu().numpy())
                ids.append(frame_ids.cpu().numpy())
    probs = np.concatenate(probs, axis = 0).flatten()
    ids = np.concatenate(ids, axis = 0).flatten()
    dataframe = []
    
    probs_list = [] 
    for id, prob in zip(ids, probs):
        dataframe.append({
            "id": int(id),
            "score": float(prob)
        })
        probs_list.append(prob) 
    submission = pd.DataFrame(dataframe)    
    submission.to_csv(f'submission-{manager.trainer_name}.csv' if manager.trainer_name else 'submission.csv', index = False)
    plot_prob(probs= probs_list, manager = manager)
    
def main():
    
    use_yaml_file = len(sys.argv) == 1+1 and '.yaml' in sys.argv[1]
    args = load_yaml_file_from_arg(sys.argv[1]) if use_yaml_file else None
    if args is None:
        raise ValueError('Please provide configuration file.')
    
    manager = get_strategy_manager(args.training_strategy)
    args.device = get_device(0) 
    evaluate_fn(args = args, manager=manager) 
    
        
if __name__  == '__main__':
    
    main()