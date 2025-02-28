import torch
import torch.optim.optimizer
import torch.utils.data.dataloader
from Dataset import VideoDataset # type: ignore
import logging
from typing import Tuple, Dict
from loss import AnticipationLoss
from model import DSA_RNN
import torch.nn as nn
from tool import AverageMeter, Monitor
import time
import os
from experiment.pseudo_Dataset import PseduoDataset
import math
import numpy as np
import torch
import random
import torch.backends.cudnn as cudnn

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training script with batch size argument')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size for training (default: 16)')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='directory to save checkpoints (default: ./checkpoints)')

    args = parser.parse_args()
    return args

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    cudnn.deterministic = True
    cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
    
def print_trainable_parameters(model):
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

def get_device()->torch.device:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA is available. Using GPU.") 
    else:
        device = torch.device('cpu')
        print(f"Device: {device}")
    return device

def get_logger() -> logging.Logger:
    logger_name = "Dashcam-Logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d %(message)s]"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger 

def get_dataloaders(val_ratio: float = 0.2) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    '''
    Return:
        train_dataloader(torch.utils.data.DataLoader)
        val_dataloader(torch.utils.data.DataLoader)
    ''' 
    if DEBUG:
        dataset = PseduoDataset()
    else:
        dataset = VideoDataset(
            root_dir="./dataset/train/extracted",
            resize_shape=(224, 224)  # Resize frames to 224x224
        )
    
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)

    split = int(np.floor(val_ratio * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True, num_workers = 4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle=False, num_workers = 4) 
    
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
        if not DEBUG:
            output, state = model(X)
        else:
            output = torch.randn(X.shape[0], 100).uniform_(0.45, 0.55)
            output.requires_grad = True
        loss = criterion(output, target) 
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step()

        batch_time.update(time.time() - start)
        start = time.time()
        current_iter = epoch * dataset_per_epoch + mini_batch_index + 1 
        remain_iter = max_iter - current_iter
        remain_time = batch_time.avg_value * remain_iter 
        loss_meter.update(loss.item())
        
        positive = torch.max(output, dim = 1).values >= 0.5
        negative = torch.max(output, dim = 1).values < 0.5
        true_case = target == 1
        false_case = target == 0
        true_meter.update(true_case.sum().item())
        false_meter.update(false_case.sum().item() )
        TP_meter.update((positive & true_case).sum().item()) 
        FP_meter.update((positive & false_case).sum().item()) 
        TN_meter.update((negative & false_case).sum().item()) 
        FN_meter.update((negative & true_case).sum().item()) 
        
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s)) 
        if (((mini_batch_index + 1) % PRINT_FREQ) == 0) or (mini_batch_index + 1 == len(train_loader)):
            
            logger.info(f'Epoch: [{epoch + 1}/{EPOCHS}][{mini_batch_index + 1}/{dataset_per_epoch}] '
                        f'Data {data_time.current_value:.1f} s ({data_time.avg_value:.1f} s) '
                        f'Batch {batch_time.current_value:.1f} s ({batch_time.avg_value:.1f} s) '
                        f'Remain(estimation) {remain_time} '
                        f'Loss {(loss_meter.current_value):.3f} ({(loss_meter.avg_value):.3f}) '
                        f'Prec {(TP_meter.current_value)/(TP_meter.current_value + FP_meter.current_value + EPS):.3f} ({(TP_meter.sum)/(TP_meter.sum + FP_meter.sum +EPS ):.3f})'
                        )
    
    
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
    for mini_batch_index, data in enumerate(val_loader):
        data_time.update(time.time() - start)
        X, target = data
        X = X.to(device)
        target = target.to(device)
        if not DEBUG:
            output, state = model(X)
        else:
            output = torch.randn(X.shape[0], 100).uniform_(0.45, 0.55)
            output.requires_grad = True
        loss = criterion(output, target) 

        batch_time.update(time.time() - start)
        start = time.time()
        current_iter = epoch * dataset_per_epoch + mini_batch_index + 1 
        remain_iter = max_iter - current_iter
        remain_time = batch_time.avg_value * remain_iter 
        loss_meter.update(loss.item())
        
        positive = torch.max(output, dim = 1).values > 0.5
        negative = torch.max(output, dim = 1).values < 0.5
        true_case = target == 1
        false_case = target == 0
        true_meter.update(true_case.sum().item())
        false_meter.update(false_case.sum().item() )
        TP_meter.update((positive & true_case).sum().item()) 
        FP_meter.update((positive & false_case).sum().item()) 
        TN_meter.update((negative & false_case).sum().item()) 
        FN_meter.update((negative & true_case).sum().item()) 
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s)) 
        if (((mini_batch_index + 1) % PRINT_FREQ) == 0) or (mini_batch_index + 1 == len(val_loader)):
            
            logger.info(f'Epoch: [{epoch + 1}/{EPOCHS}][{mini_batch_index + 1}/{dataset_per_epoch}] '
                        f'Data {data_time.current_value:.1f} s ({data_time.avg_value:.1f} s) '
                        f'Batch {batch_time.current_value:.1f} s ({batch_time.avg_value:.1f} s) '
                        f'Remain(estimation) {remain_time} '
                        f'Loss {(loss_meter.current_value):.3f} ({(loss_meter.avg_value):.3f}) '
                        f'Prec {(TP_meter.current_value)/(TP_meter.current_value + FP_meter.current_value + EPS):.3f} ({(TP_meter.sum)/(TP_meter.sum + FP_meter.sum + EPS):.3f})'
                        )
    
    
    return {'mPrec': (TP_meter.sum)/(TP_meter.sum + FP_meter.sum + EPS),
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
    DEBUG = False
    EPS = 1e-8
    set_seed(123)
    logger = get_logger()
    device = get_device()
    
    logger.info('Set-up model') 
    logger.info("=> creating model")
    
    model =  DSA_RNN(hidden_size = 64)
    model.to(device)
    
    np = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total number of parameters in model: {np}")
    print_trainable_parameters(model)
    Loss_fn = AnticipationLoss()
    optimizer = torch.optim.RAdam(model.parameters(), 
                                  betas=(0.95, 0.999), 
                                  lr = LR_RATE)
    logger.info(f'{optimizer}')
    logger.info(f'Total number of epochs: {EPOCHS}') 
    logger.info(f'Load dataset...') 
    train_dataloader, val_dataloader = get_dataloaders(val_ratio=0.2) 
    os.makedirs('model', exist_ok = True) # save model parameters under this folder
    os.makedirs('train', exist_ok = True)  # save training details under this folder
    
    tag = f'bs{BATCH_SIZE}_lr{LR_RATE}'
    monitor = Monitor(save_path = '/eos/user/y/ykao/www/kaggle/20250228/', tag = tag)
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
        valid_metrics = validation(val_loader = val_dataloader, model = model, epoch = epoch, criterion = Loss_fn)
        
        if prev_loss > valid_metrics['mLoss']:
            torch.save(model.state_dict(), f'model/best_model_ckpt_{tag}.pt')
            torch.save(optimizer.state_dict(), f'model/best_optim_ckpt_{tag}.pt')
            best_point_metrics.update(valid_metrics) 
            prev_loss = valid_metrics['mLoss']
             
        torch.save(model.state_dict(), f'model/model_ckpt-epoch{epoch:02d}_{tag}.pt')
        torch.save(optimizer.state_dict(), f'model/optim_ckpt-epoch{epoch:02d}_{tag}.pt')
        
        monitor.update(metrics = {
            'train': train_metrics,
            'validation': valid_metrics,
            'best_point': best_point_metrics
        }) 
    return



if __name__ == "__main__":
    main()    





