from utils.tool import get_device
import torch
from experiment.clip_dataset import TrainingClipDataset
import logging
from typing import Dict, Tuple
import pandas as pd
import torch.nn as nn
from utils.tool import AverageMeter, Monitor
from utils.misc import print_trainable_parameters
import time
import os
import math
import torch
import torch.nn.functional as F
from datetime import datetime
from utils.YamlArguments import load_yaml_file_from_arg
from utils.CommandLineArguments import train_parse_args
from torch.utils.data import WeightedRandomSampler
import numpy as np

from tqdm import tqdm
from models.model import get_model
from utils.optim import get_optimizer, print_optimizer_param_groups
from utils.loss import TemporalBinaryCrossEntropy
from torch.amp import autocast, GradScaler
from utils.stats import case_counting
from sklearn.metrics import average_precision_score
from utils.strategy_manager import get_strategy_manager
from utils.scheduler import get_scheduler



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
        filename = f'training-{formatted_time}.log',
        level=logging.INFO,
    )

    return logger

def get_sampler(args, metadata_df: Dict[str, pd.DataFrame]) -> Tuple[torch.utils.data.Sampler]: 

    train_sampler = WeightedRandomSampler(weights = metadata_df["train"].weight, num_samples = args.num_train_samples, replacement = False) 
    
    return train_sampler
    

def get_dataloader(args, strategy_manager, logger, epoch: int = 0 ) -> Tuple[torch.utils.data.DataLoader]:
    '''
    Create and return data loaders for training and validation.

    Args:
        args: Command line arguments
        logger: Logger object
    Returns:
        train_dataloader(torch.utils.data.DataLoader)
    '''
    # Configure augmentations based on augmentation_types argument
    print(args.augmentation_types)
    aug_config = {
        'fog': 'fog' in args.augmentation_types,
        'noise': 'noise' in args.augmentation_types,
        'gaussian_blur': 'gaussian_blur' in args.augmentation_types,
        'color_jitter': 'color_jitter' in args.augmentation_types,
        'horizontal_flip': 'horizontal_flip' in args.augmentation_types,
        'rain_effect': 'rain_effect' in args.augmentation_types,
    } if args.augmentation_types else {}

    # Log which augmentations will be used
    enabled_effects = [k for k, v in aug_config.items() if v]
    logger.info(f"Using AugmentedVideoDataset with augmentations: {enabled_effects}")
    logger.info(f"Global augmentation probability: {args.augmentation_prob}")
    if aug_config.get('horizontal_flip',False): # if key not found, use False
        logger.info(f"Horizontal flip probability: {args.horizontal_flip_prob}")

    # Use the standard dataset if augmentation is disabled
    logger.info("Using standard PreAccidentTrainDataset without augmentation")
    
    print(args.training_csv if epoch == 0 else f'{strategy_manager.monitor_dir}/frame_train-{epoch - 1}.csv') 
    #print(args.training_csv) 
    train_dataset = TrainingClipDataset(
        root_dir = args.training_dir,
        csv_file= args.training_csv if epoch == 0 else f'{strategy_manager.monitor_dir}/frame_train-{epoch - 1}.csv',
        num_frames = 16,
        frame_window = 16,
        resize_shape = (128, 171),
        crop_size = (112, 112),
        augmentation_config=aug_config,
        global_augment_prob=args.augmentation_prob,
        horizontal_flip_prob=args.horizontal_flip_prob,
    )
    val_dataset = TrainingClipDataset(
        root_dir = args.validation_dir,
        csv_file= args.validation_csv,
        num_frames = 16,
        frame_window = 16,
        resize_shape = (128, 171),
        crop_size = (112, 112),
    )
    metadata_df = {
        'train': train_dataset.metadata,
        'validation': val_dataset.metadata
    }
    train_sampler = get_sampler(args, metadata_df = metadata_df)
    # Create data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size= strategy_manager.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler = train_sampler,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size= strategy_manager.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_dataset 
    


def train(train_loader: torch.utils.data.DataLoader, model: torch.nn.Module, criterion: torch.nn.Module, cur_epoch: int, optimizer: torch.optim.Optimizer, stage_epochs: int) -> float:

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    # Metric relavant meter
    loss_meter = AverageMeter()
    bceloss_meter = AverageMeter()
    bceloss_fn = nn.CrossEntropyLoss()
    train_loss_over_iterations = []
    TP_meter = AverageMeter() #True positive
    FP_meter = AverageMeter() #False postive
    FN_meter = AverageMeter() #False negative
    TN_meter = AverageMeter() #True negative
    true_meter = AverageMeter()
    false_meter = AverageMeter()
    # Measurement
    max_iter = stage_epochs * len(train_loader)
    dataset_per_epoch = len(train_loader)
    start = time.time()
    outputs = []
    targets = []
    T_gaps = [] 
    logger.info(f'Train-procedure with {len(train_loader)} iterations')
    for mini_batch_index, data in enumerate(train_loader):
        data_time.update(time.time() - start)
        X, target, T_diff, _, concerned = data
        X = X.to(device)
        target = target.to(device)
        concerned = concerned.to(device)
        optimizer.zero_grad()
        with autocast(device_type = device.type):
            output = model(X)
            #loss = criterion(output, target)
            loss = criterion(output, target, T_diff)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10) # Gradient clip
        
        SCALER.scale(loss).backward()
        SCALER.step(optimizer)
        SCALER.update()
        ### metric 

        # Time measurement
        batch_time.update(time.time() - start)
        current_iter = cur_epoch * dataset_per_epoch + mini_batch_index + 1
        remain_iter = max_iter - current_iter
        remain_time = batch_time.avg_value * remain_iter
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if concerned.sum() > 0:
            positive, negative, true_case, false_case = case_counting(mode = 'volume', output = output[concerned], target = target[concerned])

            true_meter.update(true_case.sum().item())
            false_meter.update(false_case.sum().item() )
            TP_meter.update(torch.logical_and(positive, true_case).sum().item())
            FP_meter.update(torch.logical_and(positive, false_case).sum().item())
            TN_meter.update(torch.logical_and(negative, false_case).sum().item())
            FN_meter.update(torch.logical_and(negative, true_case).sum().item())
    
        #Metric calculation
        loss_meter.update(loss.item())
        bceloss_meter.update(bceloss_fn(output, target).item())
        train_loss_over_iterations.append(loss_meter.current_value)

        prob = F.softmax(output, dim = 1)
        outputs += prob[:, 1].tolist()
        targets += target.tolist()
        T_gaps += T_diff.tolist()
        if (((mini_batch_index + 1) % PRINT_FREQ) == 0) or (mini_batch_index + 1 == len(train_loader)):
            logger.info(f'Epoch: [{cur_epoch + 1:03d}/{stage_epochs:03d}][{mini_batch_index + 1:03d}/{dataset_per_epoch}] '
                        f'Data {data_time.current_value:.1f} s ({data_time.avg_value:.1f} s) '
                        f'Batch {batch_time.current_value:.1f} s ({batch_time.avg_value:.1f} s) '
                        f'Remain {remain_time} '
                        f'Loss[temporal-weighted] {(loss_meter.current_value):.3f} ({(loss_meter.avg_value):.3f}) '
                        f'Loss[BCE] {bceloss_meter.current_value:.3f} ({bceloss_meter.avg_value:.3f})'
                        )
        start = time.time()
        


    outputs = torch.tensor(outputs).to('cpu')
    targets = torch.tensor(targets).to('cpu')
    T_diffs = torch.tensor(T_gaps).to('cpu')
    mAP = 0
    for critical_point in [15, 30, 45]:
        mas_pos =  (T_diffs == critical_point)
        n_samples = mas_pos.sum().item() 
        neg_pool_o = outputs[~targets.bool()]
        neg_pool_t = targets[~targets.bool()] 
        rand_idx = torch.randint(0, len(neg_pool_o), (n_samples, ))
        rand_neg_o = neg_pool_o[rand_idx]
        rand_neg_t = neg_pool_t[rand_idx] 
         
        target = torch.cat([targets[mas_pos], rand_neg_t], dim = 0)
        output = torch.cat([outputs[mas_pos], rand_neg_o], dim = 0 )
        mAP += float(average_precision_score(target, output))
    mAP /=3 
    logger.info(
        f'mAP: {mAP:.3f} '
        f'mBCELoss: {bceloss_meter.avg_value:.3f} '
        f'mAcc: {(TP_meter.sum + TN_meter.sum)/(true_meter.sum + false_meter.sum + EPS):.3f} '
        f'mRecall: {TP_meter.sum/(TP_meter.sum + FN_meter.sum + EPS):.3f} '
        f'mPrec: {(TP_meter.sum)/(TP_meter.sum + FP_meter.sum  + EPS):.3f} '
        f'mSpec: {(TN_meter.sum)/(TN_meter.sum + FP_meter.sum + EPS):.3f}'
    )   
    return {'train_loss_over_iterations': train_loss_over_iterations, 
            'mLoss': loss_meter.avg_value ,
            'mBCELoss': bceloss_meter.avg_value ,
            'mAP': mAP,
            'mAcc': (TP_meter.sum + TN_meter.sum)/(true_meter.sum + false_meter.sum + EPS),
            'mRecall': TP_meter.sum/(TP_meter.sum + FN_meter.sum + EPS),
            'mPrec': (TP_meter.sum)/(TP_meter.sum + FP_meter.sum  + EPS),
            'mSpec': (TN_meter.sum)/(TN_meter.sum + FP_meter.sum + EPS),
            }
def updating_sample_weight(args, manager, train_data: torch.utils.data.dataset, model: torch.nn.Module, epoch: int):
    
    
    Loader = torch.utils.data.DataLoader(train_data, batch_size = manager.batch_size, num_workers = args.num_workers) 
    model.eval()
    
    scores = []
    truths = []
    frame_ids = [] # to store the frame index
    with torch.no_grad():
        for i, data in tqdm(enumerate(Loader), total = len(Loader)):
            X, target, T_diff, fid, concerned = data
            X = X.to(device)
            target = target.to(device)
            T_diff = T_diff.to(device)
            with autocast(device_type = device.type):
                output = model(X)
            prob = F.softmax(output, dim = 1)[..., 1] 
            scores.append(prob.unsqueeze(0).cpu())
            truths.append(target.long().unsqueeze(0).cpu())
            frame_ids.extend(list(fid))
    scores =  np.concatenate([p.flatten().cpu().numpy() for p in scores])
    truths = np.concatenate([p.flatten().cpu().numpy() for p in truths])
    fids = np.array(frame_ids)
    meta_data = pd.read_csv(args.training_csv)
    reweighting(args, meta_data, fids = fids, scores = scores, targets = truths, epoch = epoch)

def reweighting(args, manager, metadata: pd.DataFrame, fids: np.ndarray, scores: np.ndarray, targets: np.ndarray, high_confidence_threshold: float = 0.95, low_confidence_threshold: float = 0.3, epoch: int = 0) -> pd.DataFrame:
    metadata = metadata.copy()
    metadata['score'] = scores
    high_confidence_indices = np.where( ((targets == 0) & (scores <= (1 - high_confidence_threshold))) | ((targets == 1) & (scores >= (high_confidence_threshold))))[0]
    
    mask = metadata.apply(lambda row: (row.key in fids[high_confidence_indices]), axis=1)
    metadata.loc[mask, 'weight'] = 0
    
    weight_updating_regime = np.where( ((0.3<=scores) & (scores <= 0.7)))    

    mask = metadata.apply(lambda row: (row.key in fids[weight_updating_regime]), axis=1)
    #mask_pos = mask & (metadata.target == 1)
    metadata.loc[mask, 'weight'] = 10
    metadata.to_csv(f'{manager.monitor_dir}/frame_train-{epoch}.csv', index = False)

def validate(val_loader: torch.utils.data.DataLoader, model: torch.nn.Module, criterion: torch.nn.Module, cur_epoch: int, stage_epochs: int) -> float:

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
    
    max_iter = stage_epochs * len(val_loader)
    dataset_per_epoch = len(val_loader)
    start = time.time()

    logger.info(f'Validation-procedure with {len(val_loader)} iterations')
    targets = []
    outputs = []
    T_gap = []
    with torch.no_grad():
        for mini_batch_index, data in enumerate(val_loader):
            data_time.update(time.time() - start)
            X, target, T_diff, fid, concerned = data
            X = X.to(device)
            target = target.to(device)
            with autocast(device_type = device.type):
                output = model(X)
                loss = criterion(output, target, T_diff)
                bceloss_meter.update(bceloss_fn(output, target).item())
                #loss = criterion(output, target)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10) # Gradient clip
            
            if concerned.sum() > 0:
                positive, negative, true_case, false_case = case_counting(mode = 'volume', output = output[concerned], target = target[concerned])

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
            outputs += prob[:, 1].tolist()
            targets += target.tolist()
            T_gap += T_diff.tolist()
            if (mini_batch_index + 1 == len(val_loader) or mini_batch_index % PRINT_FREQ * 4 == 1):
                logger.info(f'Epoch: [{cur_epoch + 1:03d}/{stage_epochs:03d}][{mini_batch_index + 1:03d}/{dataset_per_epoch}] '
                            f'Data {data_time.current_value:.1f} s ({data_time.avg_value:.1f} s) '
                            f'Batch {batch_time.current_value:.1f} s ({batch_time.avg_value:.1f} s) '
                            f'Remain {remain_time} '
                            f'Loss[temporal-weighted] {(loss_meter.current_value):.3f} ({(loss_meter.avg_value):.3f}) '
                            f'Loss[BCE] {bceloss_meter.current_value:.3f} ({bceloss_meter.avg_value: .3f})'
                            )
            start = time.time()
            
    outputs = torch.tensor(outputs).to('cpu')
    targets = torch.tensor(targets).to('cpu')
    T_diffs = torch.tensor(T_gap).to('cpu')
    
    mAP = 0
    for critical_point in [15, 30, 45]:
        #print(T_diffs[T_diffs == critical_point])
        mas_pos =  T_diffs == critical_point
        n_samples = mas_pos.sum().item() 
        #print(targets.shape, outputs.shape) 
        neg_pool_o = outputs[~targets.bool()]
        neg_pool_t = targets[~targets.bool()] 
        rand_idx = torch.randint(0, len(neg_pool_o), (n_samples, ))
        rand_neg_o = neg_pool_o[rand_idx]
        rand_neg_t = neg_pool_t[rand_idx] 
         
        target = torch.cat([targets[mas_pos], rand_neg_t], dim = 0)
        output = torch.cat([outputs[mas_pos], rand_neg_o], dim = 0 )
        #print(average_precision_score(target, output))
        mAP += float(average_precision_score(target, output))
    mAP /=3 
       
    logger.info(
        f'mAP: {mAP:.3f} '
        f'mBCELoss:  {bceloss_meter.avg_value:.3f}'
        f'mAcc: {(TP_meter.sum + TN_meter.sum)/(true_meter.sum + false_meter.sum + EPS):.3f} '
        f'mRecall: {TP_meter.sum/(TP_meter.sum + FN_meter.sum + EPS):.3f} '
        f'mPrec: {(TP_meter.sum)/(TP_meter.sum + FP_meter.sum  + EPS):.3f} '
        f'mSpec: {(TN_meter.sum)/(TN_meter.sum + FP_meter.sum + EPS):.3f}'
    )   
        
    
    return {
        'mAP': mAP,
        'mBCELoss': bceloss_meter.avg_value ,
        'mLoss': loss_meter.avg_value ,
        'mAcc': (TP_meter.sum + TN_meter.sum)/(true_meter.sum + false_meter.sum + EPS),
        'mRecall': TP_meter.sum/(TP_meter.sum + FN_meter.sum + EPS),
        'mPrec': (TP_meter.sum)/(TP_meter.sum + FP_meter.sum  + EPS),
        'mSpec': (TN_meter.sum)/(TN_meter.sum + FP_meter.sum + EPS),
    }

def backbone_backbone_bn(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
            module.weight.requires_grad = False
            module.bias.requires_grad = False



def main():
    
    global logger, device, PRINT_FREQ,  EPS, NUM_WORKERS, RESIZE_SHAPE, SCALER,  SEED, Total_Epochs
    import sys
    use_yaml_file = len(sys.argv) == 1+1 and '.yaml' in sys.argv[1]

    args = load_yaml_file_from_arg(sys.argv[1]) if use_yaml_file else None
    if args is None:
        raise ValueError('Please provide configuration file.')
     
    SEED = args.seed
    if args.augmentation_types:
        print(f"Using augmentation types: {args.augmentation_types}")
        print(f"Global augmentation probability: {args.augmentation_prob}")
    PRINT_FREQ = args.print_freq
    EPS = 1e-8 # small number to avoid zero-division
    NUM_WORKERS = args.num_workers
    RESIZE_SHAPE = (224, 224) if args.model_type == 'swintransformer' else (112, 112) # used to set up the resize shape of frame
    logger = get_logger()
    device = get_device()

    logger.info(f'Set-up model: {args.model_type}')
    logger.info("=> creating model")

    model = get_model(model_type = args.model_type)(classifier = args.classifier)


    Total_Epochs = args.total_epochs
    
    model.to(device)
    
    manager = get_strategy_manager(args.training_strategy)
    
    Loss_fn = TemporalBinaryCrossEntropy(decay_coefficient = manager.decay_coefficient, gamma = manager.gamma) 
    #Loss_fn = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model = model,optimizer = manager.optimizer)
    scheduler = get_scheduler(scheduler = manager.scheduler, optimizer = optimizer) 
    model.freeze() 
    SCALER = GradScaler()


    os.makedirs(manager.model_dir, exist_ok = True) # save model parameters under this folder
    os.makedirs(manager.monitor_dir, exist_ok = True) # save training details under this folder

    # Generate a tag that describes the configuration
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
    train_dataloader, val_loader, train_data = get_dataloader(args = args, logger = logger, strategy_manager = manager)
    
    iterations_per_epoch = len(train_dataloader.dataset) // manager.batch_size + int(len(train_dataloader.dataset) % manager.batch_size != 0) if not args.num_train_samples else (args.num_train_samples//manager.batch_size) + int(args.num_train_samples % manager.batch_size != 0)
    
    monitor = Monitor(save_path = manager.monitor_dir, tag = tag, iterations_per_epoch = iterations_per_epoch)

    patience_counter = 0
    prev_loss = math.inf
    for cur_epoch in range(manager.epochs):
        
        for stage in manager.unfreezing['schedule']:
            stage_epoch = stage['epoch']
            if isinstance(stage_epoch, str):
                stage_epoch = int(stage_epoch)
            if cur_epoch == stage_epoch:
                print(f"Unfreezing layers: {stage['layers_to_unfreeze']}")
                model.unfreeze_layers(layers = stage['layers_to_unfreeze'])
                model.load_state_dict(torch.load(f'{manager.model_dir}/best_model_ckpt_{tag}.pt'))
                break
        
        print_optimizer_param_groups(optimizer)
        
        train_dataloader, val_loader, train_data = get_dataloader(args = args, logger = logger, strategy_manager=manager)
        
        
        logger.info('Training...')
        train_metrics = train(train_loader = train_dataloader, model = model, cur_epoch = cur_epoch, optimizer = optimizer, criterion = Loss_fn, stage_epochs = Total_Epochs)
        
        logger.info('Evaluating...')
        valid_metrics = validate(val_loader = val_loader, model = model, cur_epoch = cur_epoch, criterion = Loss_fn, stage_epochs = Total_Epochs)
        
        scheduler.step()
        if prev_loss + manager.early_stopping['delta'] > valid_metrics['mLoss']:
            torch.save(model.state_dict(), f'{manager.model_dir}/best_model_ckpt_{tag}.pt')
            #torch.save(optimizer.state_dict(), f'{manager.model_dir}/best_optim_ckpt_{tag}.pt')
            best_point_metrics.update(valid_metrics)
            best_point_metrics['current_epoch'] = cur_epoch + 1
            prev_loss = valid_metrics['mLoss']
        else:
            patience_counter  += 1
        
            
        monitor.update(metrics = {
            'train': train_metrics,
            'validation': valid_metrics,
            'best_point': best_point_metrics
        })
        torch.save(model.state_dict(), f'{manager.model_dir}/model_ckpt-epoch{cur_epoch:02d}_{tag}.pt')
        #torch.save(optimizer.state_dict(), f'{manager.model_dir}/optim_ckpt-epoch{cur_epoch:02d}_{tag}.pt')
        if patience_counter > manager.early_stopping['patience']:
            return
        
    return

if __name__ == "__main__":
    main()
