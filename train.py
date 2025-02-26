import torch
import torch.optim.optimizer
import torch.utils.data.dataloader
from Dataset import VideoDataset # type: ignore
import logging
from typing import Tuple, Dict
from loss import AnticipationLoss
from model import DSA_RNN
import torch.nn as nn
from tool import AverageMeter
import time


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
    print(torch.cuda.is_available())
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

def get_dataloader() -> torch.utils.data.DataLoader:
    dataset = VideoDataset(
        root_dir="./dataset/train/extracted",
        resize_shape=(224, 224)  # Resize frames to 224x224
    )
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4
    )
    return dataloader

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
    
    
    for mini_batch_index, data in enumerate(train_loader):
        data_time.update(time.time() - start)
        X, target = data
        if not DEBUG:
            output = model(X)
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
        
        positive = torch.max(output, dim = 1).values > 0.5
        negative = torch.max(output, dim = 1).values < 0.5
        true_case = target == 1
        false_case = target == 0
        true_meter.update(true_case.sum().item())
        false_meter.update(false_case.sum().item() )
        TP_meter.update((positive & true_case).sum().item()) 
        FP_meter.update((positive & false_case).sum().item()) 
        TN_meter.update((negative & true_case).sum().item()) 
        FN_meter.update((negative & false_case).sum().item()) 
        
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s)) 
        if ((mini_batch_index + 1) % PRINT_FREQ) == 0:
            
            logger.info(f'Epoch: [{epoch + 1}/{EPOCHS}][{mini_batch_index + 1}/{dataset_per_epoch}] '
                        f'Data {data_time.current_value:.1f} s ({data_time.avg_value:.1f} s) '
                        f'Batch {batch_time.current_value:.1f} s ({batch_time.avg_value:.1f} s) '
                        f'Remain(estimation) {remain_time} '
                        f'Loss {(loss_meter.current_value):.2f} ({(loss_meter.avg_value):.2f}) '
                        f'Acc {(true_meter.current_value)/(X.shape[0]):.2f} ({(true_meter.sum)/(true_meter.sum + false_meter.sum):.2f})'
                        )
        
        
    return {'mPrec': (TN_meter.sum + TP_meter.sum)/(TN_meter.sum + TP_meter.sum + FP_meter.sum + FN_meter.sum),
            'mRecall': TP_meter.sum/(TP_meter.sum + TN_meter.sum),
            'mLoss': loss_meter.avg_value,
            'mAcc': (true_meter.sum)/(true_meter.sum + false_meter.sum)
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
    
    
    for mini_batch_index, data in enumerate(val_loader):
        data_time.update(time.time() - start)
        X, target = data
        if not DEBUG:
            output = model(X) 
        else:
            output = torch.randn(X.shape[0], 100).uniform_(0.45, 0.55)
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
        TN_meter.update((negative & true_case).sum().item()) 
        FN_meter.update((negative & false_case).sum().item()) 
        
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s)) 
        if ((mini_batch_index + 1) % PRINT_FREQ) == 0:
            
            logger.info(f'Epoch: [{epoch + 1}/{EPOCHS}][{mini_batch_index + 1}/{dataset_per_epoch}] '
                        f'Data {data_time.current_value:.1f} s ({data_time.avg_value:.1f} s) '
                        f'Batch {batch_time.current_value:.1f} s ({batch_time.avg_value:.1f} s) '
                        f'Remain(estimation) {remain_time} '
                        f'Loss {(loss_meter.current_value):.2f} ({(loss_meter.avg_value):.2f}) '
                        f'Acc {(true_meter.current_value)/(X.shape[0]):.2f} ({(true_meter.sum)/(true_meter.sum + false_meter.sum):.2f})'
                        )
    
    
    return {'mPrec': (TN_meter.sum + TP_meter.sum)/(TN_meter.sum + TP_meter.sum + FP_meter.sum + FN_meter.sum),
            'mRecall': TP_meter.sum/(TP_meter.sum + TN_meter.sum),
            'mLoss': loss_meter.avg_value,
            'mAcc': (true_meter.sum)/(true_meter.sum + false_meter.sum)
                } 

def main():
    global logger, device, EPOCHS, PRINT_FREQ, DEBUG
    PRINT_FREQ = 16
    EPOCHS= 10
    DEBUG = True
    logger = get_logger()
    device = get_device()
    logger.info('Set-up model') 
    logger.info("=> creating model")
    model =  DSA_RNN(hidden_size = 64)
    model.to(device)
    np = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total number of parameters in model: {np}")
    #logger.info(model)
    print_trainable_parameters(model)
    Loss_fn = AnticipationLoss()
    optimizer = torch.optim.RAdam(model.parameters(), 
                                  betas=(0.95, 0.999), 
                                  lr = 0.001)
    logger.info(f'{optimizer}')
    logger.info(f'Total number of epochs: {EPOCHS}') 
    
    train_dataloader = get_dataloader() 
     
    for epoch in range(EPOCHS):
        logger.info('Training...')
        train_metrics = train(train_loader = train_dataloader, model = model, epoch = epoch, optimizer = optimizer, criterion = Loss_fn)
        valid_metric = validation(val_loader = train_dataloader, model = model, epoch = epoch, criterion = Loss_fn)
    return



if __name__ == "__main__":
    
    main()    





