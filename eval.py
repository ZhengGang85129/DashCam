import torch
import torch.nn as nn
import os
from models.model import baseline_model
from utils.tool import get_device
from utils.Dataset import VideoTo3DImageDataset
from typing import Optional, Dict
from utils.misc import parse_args
from collections import defaultdict 
import argparse
from utils.tool import AverageMeter 
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def _parse_args() -> argparse.ArgumentParser:
    parser = parse_args()
    parser.add_argument('--model_ckpt', type = str, default = None, help = 'path to your model check point.') 
    #parser.add_argument('--mode', type = str, choices = ['inference', 'evaluation'], required = True)
    parser.add_argument('--save_dir', type = str, default = './eval')
    args = parser.parse_args()  
    return args     

def get_dataloader()-> torch.utils.data.DataLoader:
    '''
    Return:
        test dataloader (torch.utils.)
    '''
    test_dataset = VideoTo3DImageDataset(
        root_dir="./dataset/train",
        csv_file = './dataset/validation_videos.csv',
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size = 16, num_workers = 4, pin_memory = True
    )
    return test_loader


def load_model(state_path: Optional[str] = None) -> nn.Module:
    model = baseline_model()
    if state_path is None:
        raise FileNotFoundError(f"The state_dict is None.")
    elif not os.path.isfile(state_path):
        raise FileNotFoundError(f"The state_dict {state_path} doesn't exist.")
    saved_state = torch.load(state_path, map_location = device if device is not None else 'cpu')

    if isinstance(saved_state, dict):
        if 'model_state_dict' in saved_state:
            model.load_state_dict(saved_state['model_state_dict'])
        else:
            model.load_state_dict(saved_state)
    else:
        model = saved_state
    
    return model

def plot_likelihood(pred:torch.Tensor, truth: torch.Tensor)-> None:
    plt.figure(figsize=(10, 6))
    plt.hist(pred[truth==1], bins=20, alpha=0.7, color='steelblue', edgecolor='black', label = 'accident')
    plt.hist(pred[truth==0], bins=20, alpha=0.7, color='red', edgecolor='black', label = 'non-accident')
    plt.title('Model output')
    plt.xlabel('Output probability')
    plt.ylabel('event')
    plt.grid(alpha=0.3)
    plt.show()
    plt.legend()
    output = os.path.join(args.save_dir, f'likelihood.png')
    plt.savefig(output)
    print(f'Check -> {output}')
    return 

def manual_precision_recall_curve(y_true, y_pred, figsize = (10, 6)) -> None:
    
    sorted_indices = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]
    
    thresholds = np.unique(y_pred_sorted)[::-1]
    
    
    precision_values = []
    recall_values = []
    
    total_positives = np.sum(y_true)
    
    for threshold in thresholds:
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        true_positives = np.sum((y_pred_binary == 1) & (y_true == 1))
        false_positives = np.sum((y_pred_binary == 1) & (y_true == 0))
        
        if true_positives + false_positives == 0:
            precision = 1.0
        else:
            precision = true_positives/ (false_positives + true_positives)

        if total_positives == 0:
            recall = 0.
        else:
            recall = true_positives/total_positives
        precision_values.append(precision)
        recall_values.append(recall)
    
    ap = 0
    
    for i in range(1, len(recall_values)):
        width = recall_values[i] - recall_values[i - 1]
        height = (precision_values[i-1] + precision_values[i])/2
        ap += width * height
    
    plt.figure(figsize = figsize)
    
    plt.plot(recall_values, precision_values, color = 'blue', lw = 2, label = f'Precision-Recall curve AP = {ap:.3f}')
    
    plt.plot([0, 1], [np.sum(y_true) / len(y_true)] * 2, 'r--', lw=2, label=f'Random (AP = {np.sum(y_true) / len(y_true):.3f})')
    
    # Set plot details
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Manual Calculation)')
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.6)
    output = os.path.join(args.save_dir, f'precision-recall.png') 
    plt.savefig(output)
      
    
def inference() -> None:
    global device, model, test_loader
    device = get_device()
    model = load_model(state_path = args.model_ckpt)
    model = model.to(device)
    test_loader = get_dataloader()
    
    batch_meter = AverageMeter()
    data_meter = AverageMeter()
     
    n_iterations = len(test_loader.dataset) // test_loader.batch_size + int(len(test_loader.dataset) % test_loader.batch_size != 0)

    
    print(f'Total number of iterations: {n_iterations}') 
    start_time = time.time()
    
    probs = []
    truth = []
     
    for batch_idx, batch in enumerate(test_loader):
        X, y = batch
        X = X.to(device)
        y = y.to(device) 
        data_meter.update(time.time() - start_time)
        print(f'Epochs: [{batch_idx + 1:03d}]/[{n_iterations:03d}]')
        print(f'Data loading time: {data_meter.current_value:.3f} sec ({data_meter.avg_value:.3f} sec)')
        
        
        start_time = time.time()
        model.eval()
        with torch.no_grad():
            output = model(X)
        prob = F.softmax(output, dim = 1)[...,1] 
        probs.append(prob.unsqueeze(0)) 
        truth.append(y.long().unsqueeze(0)) 
        batch_meter.update(time.time() - start_time)
        print(f'Inference time: {batch_meter.current_value:.2f} sec ({batch_meter.avg_value:.2f} sec)')
        start_time = time.time()
    
    probs =  np.concatenate([p.flatten().cpu().numpy() for p in probs])
    truth = np.concatenate([p.flatten().cpu().numpy() for p in truth])
    os.mkdirs(args.save_dir, exists = True)
    plot_likelihood(pred = probs, truth = truth)
    manual_precision_recall_curve(y_pred = probs, y_true = truth) 
    return
 
    
def main():
    global args 
    args = _parse_args()
    inference()
    #eval(args.mode+'()')
    
    
    
    
if __name__ == "__main__":
    main() 