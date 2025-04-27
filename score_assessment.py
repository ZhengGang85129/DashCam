import torch
import torch.nn as nn
import os
from models.model import get_model
from utils.tool import get_device
from typing import Optional, Dict, Tuple
from utils.misc import parse_args
from collections import defaultdict 
import argparse
from utils.tool import AverageMeter 
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from utils.accident_validation_dataset import PreAccidentValidationDataset 
'''
python3 ./score_assessment.py --model_ckpt <CHECKPOINT> --model_type <MODEL_TYPE> --save_dir ./model_assessment --num_workers 8 
'''
def _parse_args() -> argparse.ArgumentParser:
    parser = parse_args()
    parser.add_argument('--model_ckpt', type = str, default = None, help = 'path to your model check point.') 
    parser.add_argument('--model_type', type = str, required = True)
    parser.add_argument('--save_dir', type = str, default = './model_assessment', help = 'path to save your output result.(default: model_assessment)')
    parser.add_argument('--num_workers', type = int, default = 4, help = 'number of workers.(default: 4)')
    parser.add_argument('--tag', type = str, default = None, help = 'additional tag to fig name') 
    args = parser.parse_args()  
    return args     

def get_dataloader(root_dir: str = 'dataset/sliding_window/evaluation/tta_500ms', csv_file: str = 'dataset/sliding_window/validation_videos.csv')-> torch.utils.data.DataLoader:
    '''
    Return:
        test dataloader (torch.utils.)
    '''
    test_dataset = PreAccidentValidationDataset(
        root_dir = root_dir,
        csv_file = csv_file,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size = 16, num_workers = args.num_workers, pin_memory = True
    )
    return test_loader


def load_model(state_path: Optional[str] = None) -> nn.Module:
    model = get_model(args.model_type)()
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

def plot_likelihood(preds:torch.Tensor, truth: torch.Tensor)-> None:
    
    plt.figure(figsize=(10, 6))
    for Idx, (tag, pred) in enumerate(preds.items()):
        plt.hist(pred[truth[tag]==1], bins=20, alpha=0.7, edgecolor='black', label = tag)
        plt.hist(pred[truth[tag]==0], bins=20, alpha=0.1, color='red', edgecolor='black', label = 'non-accident' if Idx == 0 else '')
        plt.title('Model output')
        plt.xlabel('Output probability')
        plt.ylabel('event')
        plt.grid(alpha=0.3)
        plt.show()
        plt.legend()

    Tag = "" if args.tag is None else f"_{args.tag}"
    output = os.path.join(args.save_dir, f'likelihood{Tag}.png')
    plt.savefig(output)
    print('Check -> ', output)
    plt.savefig(output.replace('.png', '.pdf'))
    print('Check -> ', output.replace('.png', '.pdf')) 
    return 

def manual_precision_recall_curve(preds: Dict[str, np.array], targets: Dict[str, np.array], figsize: Tuple[int, int] = (10, 6)) -> None:
    plt.figure(figsize = figsize)
    for Idx, (tag, pred) in enumerate(preds.items()):    
        sorted_indices = np.argsort(pred)[::-1]
        y_true_sorted = targets[tag][sorted_indices]
        y_pred_sorted = pred[sorted_indices]
        
        thresholds = np.unique(y_pred_sorted)[::-1]
        
        
        precision_values = []
        recall_values = []
        
        total_positives = np.sum(targets[tag])
        
        for threshold in thresholds:
            y_pred_binary = (pred >= threshold).astype(int)
            
            true_positives = np.sum((y_pred_binary == 1) & (targets[tag] == 1))
            false_positives = np.sum((y_pred_binary == 1) & (targets[tag] == 0))
            
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
        
        
        plt.plot(recall_values, precision_values, lw = 2, label = f'Precision-Recall curve AP = {ap:.3f}')
        if Idx == 0: 
            plt.plot([0, 1], [np.sum(targets[tag]) / len(targets[tag])] * 2, 'r--', lw=2, label=f'Random (AP = {np.sum(targets[tag]) / len(targets[tag]):.3f})')
    
    # Set plot details
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Manual Calculation)')
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.6)
    Tag = "" if args.tag is None else f"_{args.tag}"
    output = os.path.join(args.save_dir, f'precision-recall{Tag}.png') 
    plt.savefig(output)
    print('Check -> ', output)
    plt.savefig(output.replace('.png', '.pdf'))
    print('Check -> ', output.replace('.png', '.pdf')) 
    
def inference(folder: str) -> None:
    print(f'Processing ({folder}) ...')
    test_loader = get_dataloader(folder)
    
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
    return probs, truth
 
    
def main():
    global args 
    args = _parse_args()
    global device, model, test_loader
    device = get_device()
    model = load_model(state_path = args.model_ckpt)
    model = model.to(device)
    model.eval()
    
    evaluation_folders = {
        'tta-500ms' : './dataset/sliding_window/evaluation/tta_500ms', 
        'tta-1000ms': './dataset/sliding_window/evaluation/tta_1000ms', 
        'tta-1500ms': './dataset/sliding_window/evaluation/tta_1500ms'
    }
    preds = dict()
    truth = dict()
    for tag, folder in evaluation_folders.items(): 
        preds[tag], truth[tag] = inference(folder)
    
    os.makedirs(args.save_dir, exist_ok=True)
    plot_likelihood(preds = preds, truth = truth)
    manual_precision_recall_curve(preds = preds, targets = truth) 
    #eval(args.mode+'()')
    
if __name__ == "__main__":
    main() 
