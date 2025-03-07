import torch
import torch.optim.optimizer
import torch.utils.data.dataloader
from utils.Dataset import VideoDataset, VideoTo3DImageDataset # type: ignore
import logging
from typing import Tuple, Dict, Union

from loss import AnticipationLoss
import torch.nn as nn
from tool import AverageMeter, Monitor
import time
import os
from experiment.pseudo_Dataset import PseduoDataset
import math
import numpy as np
import torch
from tool import get_device, set_seed
import argparse
from torchvision import transforms
import torch.nn.functional as F
from datetime import datetime
from models.model import DSA_RNN
from models.model import baseline_model
#r3d_18


def parse_args():
    parser = argparse.ArgumentParser(description='Training script with batch size argument')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size for training (default: 16)')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--model_dir', type=str, default='model',
                        help='directory to save models (default: ./model)')
    parser.add_argument('--monitor_dir', type=str, default='train',
                        help='directory to save monitoring plots (default: ./train)')
    parser.add_argument('--debug', action = "store_true", help = 'Activate to turn on the debug mode')

    args = parser.parse_args()
    return args

    
def print_trainable_parameters(model) -> None:
    """Print only the trainable parts of the model architecture"""
    logger.info("Trainable Architecture Components:")
    
    for name, module in model.named_children():
        # Check if any parameters in this module are trainable
        has_trainable = any(p.requires_grad for p in module.parameters())
        
        if has_trainable:
            logger.info(f"\n{name}:")
            # Check if it's a container module (like Sequential)
            if isinstance(module, nn.Sequential) or len(list(module.children())) > 0:
                # Print each trainable submodule
                for sub_name, sub_module in module.named_children():
                    sub_trainable = any(p.requires_grad for p in sub_module.parameters())
                    if sub_trainable:
                        logger.info(f"  {sub_name}: {sub_module}")
            else:
                # Print the module directly
                print(f"  {module}")


def get_logger() -> logging.Logger:
    logger_name = "Dashcam-Logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d %(message)s]"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    current_time = datetime.now()
    formatted_time = current_time.strftime("%m%d%H%M")
    logging.basicConfig(
        filename = f'training-lr{LR_RATE}-bs{BATCH_SIZE}-{formatted_time}.log',
        level=logging.INFO,
    )
    
    return logger 

def get_dataloaders(val_ratio: float = 0.2) -> Tuple[torch.utils.data.DataLoader, Union[torch.utils.data.DataLoader, None]]:
    '''
    Return:
        train_dataloader(torch.utils.data.DataLoader)
        val_dataloader(torch.utils.data.DataLoader)
    ''' 
    train_dataset = VideoTo3DImageDataset(
        root_dir="./dataset/train/",
        csv_file = './dataset/train_videos.csv',
    )
    if DEBUG:
        max_size = 64
        indices = list(range(len(train_dataset)))
        indices = indices[:max_size]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4, pin_memory = True)
    
    if DEBUG:
        return train_loader, None
    
    
    val_dataset = VideoDataset(
        root_dir="./dataset/train",
        csv_file = './dataset/validation_videos.csv',
    )
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers = 4, pin_memory = True) 
    return train_loader, val_loader

def train(train_loader: torch.utils.data.DataLoader, model: torch.nn.Module, criterion: torch.nn.Module, epoch: int, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    
    model.train()
    
    batch_time = AverageMeter() 
    data_time = AverageMeter()
    
    # Metric relavant meter
    loss_meter = AverageMeter()
    true_meter = AverageMeter()
    false_meter = AverageMeter()
    TP_meter = AverageMeter() #True positive
    FP_meter = AverageMeter() #False postive
    FN_meter = AverageMeter() #False negative
    TN_meter = AverageMeter() #True negative
    
    max_iter = EPOCHS * len(train_loader)
    dataset_per_epoch = len(train_loader)
    start = time.time() 
    
    logger.info(f'Train-procedure with {len(train_loader)} iterations') 
    for mini_batch_index, data in enumerate(train_loader):
        data_time.update(time.time() - start)
        X, target = data
        X = X.to(device)
        target = target.to(device)
        optimizer.zero_grad() 
        output = model(X)
        loss = criterion(output, target) 
        loss.backward() 
        optimizer.step()
        
        # Positive and Negative cases counting
        positive_probs = F.softmax(output, dim=-1)[:, 1] 
        positive = (positive_probs >= 0.5)
        positive.requires_grad = False
        negative =  ~positive
        true_case = target == 1
        false_case = ~true_case 
        
        # Time measurement 
        batch_time.update(time.time() - start)
        current_iter = epoch * dataset_per_epoch + mini_batch_index + 1 
        remain_iter = max_iter - current_iter
        remain_time = batch_time.avg_value * remain_iter 
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s)) 
        
        #Metric calculation
        loss_meter.update(loss.item())
        true_meter.update(true_case.sum().item())
        false_meter.update(false_case.sum().item() )
        TP_meter.update(torch.logical_and(positive, true_case).sum().item()) 
        FP_meter.update(torch.logical_and(positive, false_case).sum().item()) 
        TN_meter.update(torch.logical_and(negative, false_case).sum().item()) 
        FN_meter.update(torch.logical_and(negative, true_case).sum().item()) 
        
        if (((mini_batch_index + 1) % PRINT_FREQ) == 0) or (mini_batch_index + 1 == len(train_loader)):
            logger.info(f'Epoch: [{epoch + 1:03d}/{EPOCHS:03d}][{mini_batch_index + 1:03d}/{dataset_per_epoch}] '
                        f'Data {data_time.current_value:.1f} s ({data_time.avg_value:.1f} s) '
                        f'Batch {batch_time.current_value:.1f} s ({batch_time.avg_value:.1f} s) '
                        f'Remain {remain_time} '
                        f'Loss {(loss_meter.current_value):.3f} ({(loss_meter.avg_value):.3f}) '
                        f'Prec {(TP_meter.current_value)/(TP_meter.current_value + FP_meter.current_value + EPS):.3f} ({(TP_meter.sum)/(TP_meter.sum + FP_meter.sum +EPS ):.3f}) '
                        f'Spec {(TN_meter.current_value)/(TN_meter.current_value + FP_meter.current_value + EPS):.3f} ({(TN_meter.sum)/(TN_meter.sum + FP_meter.sum + EPS):.3f}) '
                        f'RCall {TP_meter.current_value/(TP_meter.current_value + FN_meter.current_value + EPS):.3f} ({TP_meter.sum/(TP_meter.sum + FN_meter.sum + EPS):.3f}) '
                        f'Acc {(TP_meter.current_value + TN_meter.current_value)/(true_meter.current_value + false_meter.current_value) + EPS:.3f} ({(TP_meter.sum + TN_meter.sum)/(true_meter.sum + false_meter.sum + EPS):.3f})'
                        )
        start = time.time()
    
    
    return {'mPrec': (TP_meter.sum)/(TP_meter.sum + FP_meter.sum  + EPS),
            'mRecall': TP_meter.sum/(TP_meter.sum + FN_meter.sum + EPS),
            'mLoss': loss_meter.avg_value,
            'mAcc': (TP_meter.sum + TN_meter.sum)/(true_meter.sum + false_meter.sum + EPS)
                } 


def validation(val_loader: torch.utils.data.DataLoader, model: torch.nn.Module, criterion: torch.nn.Module, epoch: int)-> Dict[str, float]:
    
    model.eval()
    
    batch_time = AverageMeter() 
    data_time = AverageMeter()
    
    # Metric relavant meter
    loss_meter = AverageMeter()
    true_meter = AverageMeter()
    false_meter = AverageMeter()
    TP_meter = AverageMeter() #True positive
    FP_meter = AverageMeter() #False postive
    FN_meter = AverageMeter() #False negative
    TN_meter = AverageMeter() #True negative
    
    max_iter = EPOCHS * len(val_loader)
    dataset_per_epoch = len(val_loader)
    start = time.time() 
    
    
    logger.info(f'Validation-procedure with {len(val_loader)} iterations') 
    with torch.no_grad():
        for mini_batch_index, data in enumerate(val_loader):
            data_time.update(time.time() - start)
            X, target = data
            X = X.to(device)
            target = target.to(device)
            output = model(X)
            loss = criterion(output, target) 

            # Positive and Negative cases counting
            positive_probs = F.softmax(output, dim=-1)[:, 1] 
            positive = (positive_probs >= 0.5)
            positive.requires_grad = False
            negative =  ~positive
            true_case = target == 1
            false_case = ~true_case 
            
            # Time measurement 
            batch_time.update(time.time() - start)
            current_iter = epoch * dataset_per_epoch + mini_batch_index + 1 
            remain_iter = max_iter - current_iter
            remain_time = batch_time.avg_value * remain_iter 
            t_m, t_s = divmod(remain_time, 60)
            t_h, t_m = divmod(t_m, 60)
            remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s)) 
            
            #Metric calculation
            loss_meter.update(loss.item())
            true_meter.update(true_case.sum().item())
            false_meter.update(false_case.sum().item() )
            TP_meter.update(torch.logical_and(positive, true_case).sum().item()) 
            FP_meter.update(torch.logical_and(positive, false_case).sum().item()) 
            TN_meter.update(torch.logical_and(negative, false_case).sum().item()) 
            FN_meter.update(torch.logical_and(negative, true_case).sum().item()) 
            
            if (((mini_batch_index + 1) % PRINT_FREQ) == 0) or (mini_batch_index + 1 == len(train_loader)):
                logger.info(f'Epoch: [{epoch + 1:03d}/{EPOCHS:03d}][{mini_batch_index + 1:03d}/{dataset_per_epoch}] '
                            f'Data {data_time.current_value:.1f} s ({data_time.avg_value:.1f} s) '
                            f'Batch {batch_time.current_value:.1f} s ({batch_time.avg_value:.1f} s) '
                            f'Remain {remain_time} '
                            f'Loss {(loss_meter.current_value):.3f} ({(loss_meter.avg_value):.3f}) '
                            f'Prec {(TP_meter.current_value)/(TP_meter.current_value + FP_meter.current_value + EPS):.3f} ({(TP_meter.sum)/(TP_meter.sum + FP_meter.sum +EPS ):.3f}) '
                            f'Spec {(TN_meter.current_value)/(TN_meter.current_value + FP_meter.current_value + EPS):.3f} ({(TN_meter.sum)/(TN_meter.sum + FP_meter.sum + EPS):.3f}) '
                            f'RCall {TP_meter.current_value/(TP_meter.current_value + FN_meter.current_value + EPS):.3f} ({TP_meter.sum/(TP_meter.sum + FN_meter.sum + EPS):.3f}) '
                            f'Acc {(TP_meter.current_value + TN_meter.current_value)/(true_meter.current_value + false_meter.current_value) + EPS:.3f} ({(TP_meter.sum + TN_meter.sum)/(true_meter.sum + false_meter.sum + EPS):.3f})'
                            )
            start = time.time()
        
    
    return {'mPrec': (TP_meter.sum)/(TP_meter.sum + FP_meter.sum  + EPS),
            'mRecall': TP_meter.sum/(TP_meter.sum + FN_meter.sum + EPS),
            'mLoss': loss_meter.avg_value,
            'mAcc': (TP_meter.sum + TN_meter.sum)/(true_meter.sum + false_meter.sum + EPS)
                } 
            

def main():
    global logger, device, EPOCHS, PRINT_FREQ, DEBUG, LR_RATE, BATCH_SIZE, EPS

    args = parse_args()
    print(f"Training with batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Number of epochs: {args.epochs}")

    BATCH_SIZE = args.batch_size
    PRINT_FREQ = 4
    EPOCHS= args.epochs
    LR_RATE = args.learning_rate
    DEBUG = args.debug
    EPS = 1e-8
    #DECAY_NFRAME = 20
    set_seed(123)
    logger = get_logger()
    device = get_device()
    
    logger.info('Set-up model') 
    logger.info("=> creating model")
    
    
    model = baseline_model()
     
    model.to(device)
    
    np = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total number of parameters in model: {np}")
    print_trainable_parameters(model)
    Loss_fn = nn.CrossEntropyLoss()
    #AnticipationLoss(decay_nframe = DECAY_NFRAME, pivot_frame_index = 100, device = get_device())
    
    optimizer = torch.optim.RAdam([p for p in model.parameters() if p.requires_grad], lr = LR_RATE)
    logger.info(f'{optimizer}')
    logger.info(f'Total number of epochs: {EPOCHS}') 
    logger.info(f'Load dataset...') 
    train_dataloader, val_dataloader = get_dataloaders(val_ratio=0.2) 
    
    os.makedirs(args.model_dir, exist_ok = True) # save model parameters under this folder
    os.makedirs(args.monitor_dir, exist_ok = True) # save training details under this folder
    
    tag = f'bs{BATCH_SIZE}_lr{LR_RATE}'
    monitor = Monitor(save_path = args.monitor_dir, tag = tag)
    best_point_metrics = {
        'mLoss': float('inf'),
        'mPrec': -float('inf'),
        'mRecall': -float('inf'),
        'mAcc': -float('inf'),
        'current_epoch': 0
    }
    
    prev_loss = math.inf
     
    for epoch in range(EPOCHS):
        logger.info('Training...')
        train_metrics = train(train_loader = train_dataloader, model = model, epoch = epoch, optimizer = optimizer, criterion = Loss_fn)
        logger.info('Evaluating...')
        if not DEBUG:
            valid_metrics = validation(val_loader = val_dataloader, model = model, epoch = epoch, criterion = Loss_fn)
            
            if prev_loss > valid_metrics['mLoss']:
                torch.save(model.state_dict(), f'{args.model_dir}/best_model_ckpt_{tag}.pt')
                torch.save(optimizer.state_dict(), f'{args.model_dir}/best_optim_ckpt_{tag}.pt')
                best_point_metrics.update(valid_metrics) 
                best_point_metrics['current_epoch'] = epoch 
                prev_loss = valid_metrics['mLoss']
             
            monitor.update(metrics = {
                'train': train_metrics,
                'validation': valid_metrics,
                'best_point': best_point_metrics
            }) 
        
        torch.save(model.state_dict(), f'{args.model_dir}/model_ckpt-epoch{epoch:02d}_{tag}.pt')
        torch.save(optimizer.state_dict(), f'{args.model_dir}/optim_ckpt-epoch{epoch:02d}_{tag}.pt')
    return



if __name__ == "__main__":
    main()    





