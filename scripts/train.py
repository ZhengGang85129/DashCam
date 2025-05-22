import torch
from src.datasets.accident_dataset import AccidentDataset
from utils.tool import get_device
from models.baseline_model import LayerNorm3D
import logging
from typing import  Tuple
import pandas as pd
import torch.nn as nn
from utils.tool import AverageMeter
import time
import os
import math
import torch
import torch.nn.functional as F
from datetime import datetime
from utils.YamlArguments import load_yaml_file_from_arg
from torch.utils.data import WeightedRandomSampler
import numpy as np
import mlflow
import mlflow.pytorch
from tqdm import tqdm
from models.model import get_model
from utils.optim import get_optimizer 
from utils.loss import TemporalBinaryCrossEntropy 
from torch.cuda.amp import autocast, GradScaler
from utils.stats import case_counting
from sklearn.metrics import average_precision_score
from utils.strategy_manager import get_strategy_manager
from utils.scheduler import get_scheduler
import sys
from mlflow.models.signature import infer_signature

conda_env = {
    'channels': ['defaults', 'conda-forge'],
    'dependencies': [
        'python=3.9',
        'torch==2.0.1+cu117',
        'torchvision==0.15.2+cu117',  
        'pip',
        {
            'mlflow', 
        }
    ]
}


def get_logger() -> logging.Logger:
    logger_name = "Dashcam-Logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d %(message)s]"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    current_time = datetime.now()
    formatted_time = current_time.strftime("%s%m%d%H%M")
    logging.basicConfig(
        filename = f'log/training-{formatted_time}.log',
        level=logging.INFO,
    )

    return logger

def get_sampler(args, metadata_df: pd.DataFrame) -> Tuple[torch.utils.data.Sampler]: 
    """
    Args:
        args: argparse.Namespace containing configuration parameters.
        metadata_df (pd.DataFrame): 
            - DataFrame of metadata containing:
                - `clip_id` (str): image file name
                - video_id: from which video
                - target: positive or negative 
                - weight: weight of sample
                - T_diff: frame difference to accident of frame
                - frame: frame index in the video
                - key: key to get the row
                - accident_frame: frame of accident in the video
                - concerned: whether the frame is the frame before accident with 500/1000/1500 ms
                 
    """
    train_sampler = WeightedRandomSampler(weights = metadata_df.weight, num_samples = args.num_train_samples, replacement = False) 
    
    return train_sampler

def create_dataloader(args, manager, logger, metadata_path: str, data_type: str = 'train')  -> Tuple[torch.utils.data.DataLoader, torch.utils.data.dataset]:  
    """
    Args:
        args: argparse.Namespace containing configuration parameters.
        manager: Manager object containing training details.
        metadata_path (str): Path to metadata CSV file.
        data_type (str): Either 'train' or 'validation'
    Returns:
        Tuple of (DataLoader, Dataset).
    """ 
    aug_config = {}
    augmentation_prob = args.augmentation_prob if data_type == 'train' else 0
    horizontal_flip_prob = args.horizontal_flip_prob if data_type == 'train' else 0
    root_dir = args.training_dir if data_type == 'train' else args.validation_dir
    if data_type == 'train':
        if args.augmentation_types :
            aug_config = {
                'fog': 'fog' in args.augmentation_types,
                'noise': 'noise' in args.augmentation_types,
                'gaussian_blur': 'gaussian_blur' in args.augmentation_types,
                'color_jitter': 'color_jitter' in args.augmentation_types,
                'horizontal_flip': 'horizontal_flip' in args.augmentation_types,
                'rain_effect': 'rain_effect' in args.augmentation_types,
            } 
            # Log which augmentations will be used
            enabled_effects = [k for k, v in aug_config.items() if v]
            logger.info(f"Using AugmentedVideoDataset with augmentations: {enabled_effects}")
            logger.info(f"Global augmentation probability: {args.augmentation_prob}")
            if aug_config.get('horizontal_flip',False): # if key not found, use False
                logger.info(f"Horizontal flip probability: {args.horizontal_flip_prob}")
            # Use the standard dataset if augmentation is disabled
            logger.info(f"Using standard PreAccidentTrainDataset without augmentation({data_type})")
        else:
            logger.info(f"No augmentation is specified({data_type})") 
        #print(args.training_csv) 
    # --------------------
    # Dataset construction
    # --------------------
    
    dataset = AccidentDataset(
        root_dir = root_dir,
        csv_file= metadata_path,
        mode = 'training' if data_type == 'train' else 'validation',
        model_type = args.model_type,
        stride = manager.stride ,
        frame_per_window = 16,
    )
    
    # --------------------------------
    # DataLoader with optional sampler
    # --------------------------------
    if data_type == 'train': 
        #train_sampler = get_sampler(args, metadata_df = dataset.metadata)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size= manager.batch_size,
            num_workers = args.num_workers,
            pin_memory = True,
            #sampler = train_sampler,
        )
    elif data_type == 'validation': 
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size= manager.batch_size,
            num_workers = args.num_workers,
            pin_memory = True,
        )
    else:
        raise ValueError(f'Data type: {data_type} is not supported.')
    return loader, dataset 

import subprocess

def get_gpu_usage():
    result = subprocess.run(
        ['nvidia-smi', "--query-gpu=utilization.gpu,memory.used,memory.free,memory.total", '--format=csv,noheader,nounits'],
        stdout = subprocess.PIPE,
        text = True
    )
    gpu_usage = result.stdout.strip().split("\n")[0].split(",")[0]
    return float(gpu_usage)

def train(args, logger, train_loader: torch.utils.data.DataLoader, model: torch.nn.Module, criterion: torch.nn.Module, cur_epoch: int, optimizer: torch.optim.Optimizer, device: torch.cuda.device, scaler:torch.cuda.amp.grad_scaler) -> float:

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    # Metric relavant meter
    loss_meter = AverageMeter()
    #bceloss_meter = AverageMeter()
    #bceloss_fn = nn.CrossEntropyLoss()
    train_loss_over_iterations = []
    TP_meter = AverageMeter() #True positive
    FP_meter = AverageMeter() #False postive
    FN_meter = AverageMeter() #False negative
    TN_meter = AverageMeter() #True negative
    true_meter = AverageMeter()
    false_meter = AverageMeter()
    # Measurement
    max_iter = args.total_epochs * len(train_loader)
    dataset_per_epoch = len(train_loader)
    start = time.time()
    all_y_pred = []
    all_y_label = []
    T_gaps = [] 
    logger.info(f'Train-procedure with {len(train_loader)} iterations')
    N_pos_sum = 0
    N_neg_sum = 0
    for mini_batch_index, data in enumerate(train_loader):
    
        data_time.update(time.time() - start)
        X, target, T_diff, _, concerned = data
        N_pos = (target == 1).sum().item()
        N_neg = (target == 0).sum().item() 
        X = X.to(device)
        target = target.to(device)
        concerned = concerned.to(device)
        optimizer.zero_grad()
        N_pos_sum += N_pos
        N_neg_sum += N_neg
        with autocast():
            output = model(X)
            #bceloss = bceloss_fn(output, target)
            #loss = criterion(output, target, T_diff)
            loss = criterion(output, target)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5) # Gradient clip
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        ### metric 

        # Time measurement
        batch_time.update(time.time() - start)
        current_iter = cur_epoch * dataset_per_epoch + mini_batch_index + 1
        remain_iter = max_iter - current_iter
        remain_time = batch_time.avg_value * remain_iter
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        positive, negative, true_case, false_case = case_counting(mode = 'volume', output = output, target = target)
        true_meter.update(true_case.sum().item())
        false_meter.update(false_case.sum().item() )
        TP_meter.update(torch.logical_and(positive, true_case).sum().item())
        FP_meter.update(torch.logical_and(positive, false_case).sum().item())
        TN_meter.update(torch.logical_and(negative, false_case).sum().item())
        FN_meter.update(torch.logical_and(negative, true_case).sum().item())
    
        #Metric calculation
        loss_meter.update(loss.item())
        #bceloss_meter.update(bceloss.item())
        train_loss_over_iterations.append(loss_meter.current_value)

        prob = F.softmax(output, dim = 1)
        all_y_pred.append(prob.detach().cpu().numpy())
        all_y_label.append(target.cpu().numpy())
        
        T_gaps += T_diff.tolist()
        if (((mini_batch_index + 1) % args.print_freq) == 0) or (mini_batch_index + 1 == len(train_loader)):
            logger.info(f'Epoch: [{cur_epoch + 1:03d}/{args.total_epochs:03d}][{mini_batch_index + 1:03d}/{dataset_per_epoch}] '
                        f'Data {data_time.current_value:.1f} s ({data_time.avg_value:.1f} s) '
                        f'Batch {batch_time.current_value:.1f} s ({batch_time.avg_value:.1f} s) '
                        f'Remain {remain_time} '
                        f'Loss[temporal-weighted] {(loss_meter.current_value):.3f} ({(loss_meter.avg_value):.3f}) '
                        #f'Loss[BCE] {bceloss_meter.current_value:.3f} ({bceloss_meter.avg_value:.3f})'
                        )
        start = time.time()
    all_y_label = np.concatenate(np.array(all_y_label), axis = 0)  
    all_y_pred = np.concatenate(np.array(all_y_pred), axis = 0)
    
    
    mAP = float(average_precision_score(all_y_label, all_y_pred[:, 1]))
   
    
    logger.info(f'POS(sum): {N_pos_sum}, Neg(sum): {N_neg_sum}')
    logger.info(
        f'mAP: {mAP:.3f} '
        #f'mBCELoss: {bceloss_meter.avg_value:.3f} '
        f'mAcc: {(TP_meter.sum + TN_meter.sum)/(true_meter.sum + false_meter.sum + 1e-8):.3f} '
        f'mRecall: {TP_meter.sum/(TP_meter.sum + FN_meter.sum + 1e-8):.3f} '
        f'mPrec: {(TP_meter.sum)/(TP_meter.sum + FP_meter.sum  + 1e-8):.3f} '
        f'mSpec: {(TN_meter.sum)/(TN_meter.sum + FP_meter.sum + 1e-8):.3f}'
    )   
    return {'train_loss_over_iterations': train_loss_over_iterations, 
            'mLoss': loss_meter.avg_value ,
            #'mBCELoss': bceloss_meter.avg_value ,
            'mAP': mAP,
            'mAcc': (TP_meter.sum + TN_meter.sum)/(true_meter.sum + false_meter.sum + 1e-8),
            'mRecall': TP_meter.sum/(TP_meter.sum + FN_meter.sum + 1e-8),
            'mPrec': (TP_meter.sum)/(TP_meter.sum + FP_meter.sum  + 1e-8),
            'mSpec': (TN_meter.sum)/(TN_meter.sum + FP_meter.sum + 1e-8),
            }
def updating_sample_weight(args, manager, train_data: torch.utils.data.dataset, model: torch.nn.Module, epoch: int, tag: str = '', device: torch.cuda.device = None):
    
    
    Loader = torch.utils.data.DataLoader(train_data, 
                                         batch_size = manager.batch_size, 
                                         num_workers = args.num_workers,
                                         ) 
    model.eval()
    
    scores = []
    truths = []
    frame_ids = [] # to store the frame index
    with torch.no_grad():
        for i, data in tqdm(enumerate(Loader), total = len(Loader)):
            X, target, _, fid, *_  = data
            X = X.to(device)
            target = target.to(device)
            with autocast():
                output = model(X)
            prob = F.softmax(output, dim = 1)[..., 1] 
            scores.append(prob.cpu())
            truths.append(target.long().cpu())
            frame_ids.extend(list(fid))
    scores =  np.concatenate([p.cpu().numpy() for p in scores])
    truths = np.concatenate([p.cpu().numpy() for p in truths])
    fids = np.array(frame_ids)
    meta_data = pd.read_csv(args.training_csv)
    score_df = pd.DataFrame({"key": fids,
                             "score": scores,
                             "target": truths})
    meta_data = pd.merge(meta_data, score_df[['key', 'score']], on = 'key', how = 'left')
    
    assert meta_data['key'].is_unique, "Original metadata contains duplicated keys!"
    return get_reweighting_csv(manager, meta_data,  reweight_epoch = epoch, tag = tag)

def get_reweighting_csv(manager, metadata: pd.DataFrame, reweight_epoch: int = 0, tag: str = '') -> str:
    metadata = metadata.copy()
    assert 'score' in metadata.columns, "`score` column is missing." 
    assert 'target' in metadata.columns, "`target` column is missing." 
    assert metadata['score'].between(0, 1).all(), '`score` must be in [0, 1]'
    
     
    
    keep_pos_mask = (metadata.target == 1) & (metadata.score < 0.95)
    keep_neg_mask = (metadata.target == 0) & (metadata.score > 0.05)
    hard_neg_mask = (metadata.target == 0) & (metadata.score > 0.5) 
    drop_neg_mask = (metadata.target == 0) & (metadata.score < 0.05)
    #metadata['weight'] = 0.0
    #metadata.loc[keep_pos_mask, 'weight'] = (1.0 - metadata.loc[keep_pos_mask, 'score']) ** 2
    metadata.loc[hard_neg_mask, 'weight'] = np.maximum(metadata.loc[hard_neg_mask, 'score'] * 1.2, 10) 
    metadata.loc[drop_neg_mask, 'weight'] = 0
    metadata_path = f'{manager.monitor_dir}/reweight_train-{reweight_epoch}{tag}.csv'
    metadata.to_csv(metadata_path, index = False)

    return metadata_path

def validate(args, logger, val_loader: torch.utils.data.DataLoader, model: torch.nn.Module, criterion: torch.nn.Module, cur_epoch: int, device: torch.cuda.device) -> float:

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    # Metric relavant meter
 
    loss_meter = AverageMeter()
    bceloss_meter = AverageMeter()
    bceloss_fn = nn.CrossEntropyLoss()
    TP_meter = AverageMeter() #True positive
    FP_meter = AverageMeter() #False postive
    FN_meter = AverageMeter() #False negative
    TN_meter = AverageMeter() #True negative
    true_meter = AverageMeter()
    false_meter = AverageMeter()
    
    max_iter = args.total_epochs * len(val_loader)
    dataset_per_epoch = len(val_loader)
    start = time.time()

    logger.info(f'Validation-procedure with {len(val_loader)} iterations')
    all_y_label = []
    all_y_pred = []
    with torch.no_grad():
        for mini_batch_index, data in enumerate(val_loader):
            data_time.update(time.time() - start)
            X, target, T_diff, fid, concerned = data
            X = X.to(device)
            target = target.to(device)
            with autocast():
                output = model(X)
                #loss = criterion(output, target, T_diff)
                loss = criterion(output, target)
                bceloss_meter.update(bceloss_fn(output, target).item())
                #loss = criterion(output, target)
            
            positive, negative, true_case, false_case = case_counting(mode = 'volume', output = output, target = target)

            true_meter.update(true_case.sum().item())
            false_meter.update(false_case.sum().item() )
            TP_meter.update(torch.logical_and(positive, true_case).sum().item())
            FP_meter.update(torch.logical_and(positive, false_case).sum().item())
            TN_meter.update(torch.logical_and(negative, false_case).sum().item())
            FN_meter.update(torch.logical_and(negative, true_case).sum().item())

            # Time measurement
            batch_time.update(time.time() - start)
            current_iter = cur_epoch * dataset_per_epoch + mini_batch_index + 1
            remain_iter = max_iter - current_iter
            remain_time = batch_time.avg_value * remain_iter
            t_m, t_s = divmod(remain_time, 60)
            t_h, t_m = divmod(t_m, 60)
            remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

            #Metric calculation
            loss_meter.update(loss.item())

            prob = F.softmax(output, dim = 1)
            all_y_label.append(target.cpu().numpy())
            all_y_pred.append(prob.cpu().numpy())
            if (mini_batch_index + 1 == len(val_loader) or mini_batch_index % args.print_freq * 4 == 1):
                logger.info(f'Epoch: [{cur_epoch + 1:03d}/{args.total_epochs:03d}][{mini_batch_index + 1:03d}/{dataset_per_epoch}] '
                            f'Data {data_time.current_value:.1f} s ({data_time.avg_value:.1f} s) '
                            f'Batch {batch_time.current_value:.1f} s ({batch_time.avg_value:.1f} s) '
                            f'Remain {remain_time} '
                            f'Loss[temporal-weighted] {(loss_meter.current_value):.3f} ({(loss_meter.avg_value):.3f}) '
                            f'Loss[BCE] {bceloss_meter.current_value:.3f} ({bceloss_meter.avg_value: .3f})'
                            )
            start = time.time()
            
    all_y_label = np.concatenate(all_y_label, axis = 0)
    all_y_pred = np.concatenate(all_y_pred, axis = 0)
    mAP = float(average_precision_score(all_y_label, all_y_pred[:, 1]))
       
    logger.info(
        f'mAP: {mAP:.3f} '
        f'mBCELoss:  {bceloss_meter.avg_value:.3f}'
        f'mAcc: {(TP_meter.sum + TN_meter.sum)/(true_meter.sum + false_meter.sum + 1e-8):.3f} '
        f'mRecall: {TP_meter.sum/(TP_meter.sum + FN_meter.sum + 1e-8):.3f} '
        f'mPrec: {(TP_meter.sum)/(TP_meter.sum + FP_meter.sum  + 1e-8):.3f} '
        f'mSpec: {(TN_meter.sum)/(TN_meter.sum + FP_meter.sum + 1e-8):.3f}'
    )   
        
    
    return {
        'mAP': mAP,
        'mBCELoss': bceloss_meter.avg_value ,
        'mLoss': loss_meter.avg_value ,
        'mAcc': (TP_meter.sum + TN_meter.sum)/(true_meter.sum + false_meter.sum + 1e-8),
        'mRecall': TP_meter.sum/(TP_meter.sum + FN_meter.sum + 1e-8),
        'mPrec': (TP_meter.sum)/(TP_meter.sum + FP_meter.sum  + 1e-8),
        'mSpec': (TN_meter.sum)/(TN_meter.sum + FP_meter.sum + 1e-8),
    }

def unfreeze_layernorm(module):
    
    for name, child in module.named_children():
        
        if isinstance(child, LayerNorm3D):
            for param in child.parameters():
                param.requires_grad = True
        else:
            unfreeze_layernorm(child)

def freeze_batchnorm(module):
    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm2d, nn.BatchNorm3d)):
            child.weight.requires_grad = False
            child.bias.requires_grad = False
            child.running_mean.requires_grad = False
            child.running_var.requires_grad = False
            #for param in child.parameters():
            #    param.requires_grad = False
    


        


def train_fn(args, manager, logger, device):
    #mlflow.set_tracking_uri("https://localhost:5000")
    
    with mlflow.start_run() as run:
        logger.info(f'Set-up model: {args.model_type}')
        logger.info("=> creating model")
        model = get_model(model_type = args.model_type)(classifier = args.classifier)
        model.to(device)
        #Loss_fn = TemporalBinaryCrossEntropy(decay_coefficient = manager.decay_coefficient, gamma = manager.gamma) 
        #Loss_fn = FocalLoss(gamma = manager.gamma)
        criterion = nn.CrossEntropyLoss()

        optimizer = get_optimizer(model = model,optimizer = manager.optimizer)
        scheduler = get_scheduler(scheduler = args.scheduler, optimizer = optimizer) 

        os.makedirs(manager.model_dir, exist_ok = True) # save model parameters under this folder
        os.makedirs(manager.monitor_dir, exist_ok = True) # save training details under this folder
        
        if args.augmentation_types:
            # Create a shorter tag for augmentation types
            aug_types_str = '_'.join([t[:3] for t in sorted(args.augmentation_types)])
            aug_tag = f'_aug{args.augmentation_prob:.2f}_{aug_types_str}'
        else:
            aug_tag = ''

        tag = f'{manager.trainer_name}{aug_tag}'

        # Log the tag being used
        logger.info(f"Using tag: {tag}")


        best_point_metrics = {
            'mLoss': float('inf'),
            'mPrec': 0,
            'mRecall': 0,
            'mAcc': 0,
            'mAP': 0,
            'mSpec': 0, 
            'current_epoch': 0
        }
        
        train_dataloader, train_data = create_dataloader(args = args, manager = manager, logger = logger, metadata_path = args.training_csv, data_type = 'train')
        val_loader, _ = create_dataloader(args = args, manager = manager, logger = logger, metadata_path = args.validation_csv, data_type = 'validation')
        

        model = model.to('cpu')
        
        input_example = torch.rand(1, 16, 3, 112, 112) if args.model_type == 'baseline' else torch.rand(1, 16, 3, 224, 224)
        
        output_example = model(input_example).cpu().detach().numpy()
        
        signature = infer_signature(input_example.cpu().detach().numpy(), output_example)
        
        
        
        
        model = model.to(device)
        
        model.freeze() 
        scaler = GradScaler()
        
        mlflow.log_param("batch_size", manager.batch_size)
        mlflow.log_param("epochs", args.total_epochs) 
        mlflow.log_param("classifier", args.classifier)       
        mlflow.log_param("classifier_lr", manager.optimizer['differential_lr']['classifier']) 
        
        mlflow.log_param("stride", manager.stride) 
        mlflow.log_param("early_stop_patience", manager.early_stopping['patience'])
        mlflow.log_param("delta", manager.early_stopping['delta']) 
        mlflow.log_param("unfreezing_strategy", manager.unfreezing) 
        
        patience_counter = 0
        prev_loss = math.inf
        
        for cur_epoch in range(args.total_epochs):
            for stage in manager.unfreezing['schedule']:
                stage_epoch = stage['epoch']
                if isinstance(stage_epoch, str):
                    stage_epoch = int(stage_epoch)
                if cur_epoch == stage_epoch:
                    logger.info(f"Unfreezing layers: {stage['layers_to_unfreeze']}")
                    model.unfreeze_layers(layer_names = stage['layers_to_unfreeze'])
                    unfreeze_layernorm(model) 
                    freeze_batchnorm(model) 
                    break
            
            for idx, group in enumerate(optimizer.param_groups):
                lr = group['lr']
                mlflow.log_metric(f"Group-{idx}-lr", lr, step = cur_epoch)
            
            #print_optimizer_param_groups(optimizer)
            gpu_usage = get_gpu_usage() 
            mlflow.log_metric("gpu_usage", gpu_usage, step = cur_epoch)
            logger.info('Training...')
            
            train_metrics = train(args = args, logger = logger, train_loader = train_dataloader, model = model, cur_epoch = cur_epoch, optimizer = optimizer, criterion = criterion, device = device, scaler = scaler)
            
            for metric_name, metric_value in train_metrics.items():
                if metric_name == 'train_loss_over_iterations':
                    for idx, value in enumerate(metric_value):
                        mlflow.log_metric(metric_name, value, step = cur_epoch * len(train_dataloader) + idx)
                     
                    continue
                else:
                    mlflow.log_metric('train_'+metric_name, metric_value, step = cur_epoch)
             
            logger.info('Evaluating...')
            valid_metrics = validate(args = args, logger = logger, val_loader = val_loader, model = model, cur_epoch = cur_epoch, criterion = criterion, device = device)
            for metric_name, metric_value in valid_metrics.items():
                mlflow.log_metric('val_'+metric_name, metric_value, step = cur_epoch)
            
            scheduler.step()
            if prev_loss + manager.early_stopping['delta'] > valid_metrics['mLoss']:
                torch.save(model.state_dict(), f'{manager.model_dir}/best_model_ckpt_{tag}.pt')
                best_point_metrics.update(valid_metrics)
                best_point_metrics['current_epoch'] = cur_epoch + 1 
                prev_loss = valid_metrics['mLoss']
                patience_counter = 0
                mlflow.pytorch.log_model(model, f"model/{args.model_type}", conda_env = conda_env, registered_model_name = args.model_type, signature = signature, input_example = input_example.cpu().detach().numpy())
            else:
                patience_counter  += 1
            
                
            #monitor.update(metrics = {
            #    'train': train_metrics,
            #    'validation': valid_metrics,
            #    'best_point': best_point_metrics
            #})
            #torch.save(model.state_dict(), f'{manager.model_dir}/model_ckpt-epoch{cur_epoch + 1:02d}_{tag}.pt')
            gpu_usage = get_gpu_usage() 
            mlflow.log_metric("final_gpu_usage", gpu_usage, step = cur_epoch)
            if patience_counter > manager.early_stopping['patience']:
                return
            '''
            if manager.reweight and manager.reweight['apply']:
                for reweight_epoch in manager.reweight['epoch']:    
                    if cur_epoch == reweight_epoch:
                        logger.info('Start to reweight the samples.')
                        reweight_csv = updating_sample_weight(args, manager, train_data = train_data, model = model, epoch = reweight_epoch, tag = tag) 
                        logger.info(f'Check {reweight_csv}')
                        train_dataloader, train_data = create_dataloader(args = args, manager = manager, logger = logger, metadata_path = reweight_csv, data_type = 'train')
                        
                        continue
                
            ''' 
    return best_point_metrics['mLoss']


        
         
def main():
    use_yaml_file = len(sys.argv) == 1+1 and '.yaml' in sys.argv[1]
    args = load_yaml_file_from_arg(sys.argv[1]) if use_yaml_file else None
    if args is None:
        raise ValueError('Please provide configuration file.')
     
    if args.augmentation_types:
        print(f"Using augmentation types: {args.augmentation_types}")
        print(f"Global augmentation probability: {args.augmentation_prob}")
    
    
    os.makedirs('log', exist_ok = True)
    logger = get_logger()
    device = get_device()
    manager = get_strategy_manager(args.training_strategy)
    loss = train_fn(args = args, manager = manager, logger = logger, device = device)


if __name__ == "__main__":
    main()
