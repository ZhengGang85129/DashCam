import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
import torch.utils.data.dataloader
from utils.Dataset import VideoDataset, VideoTo3DImageDataset # type: ignore
from utils.accident_validation_dataset import PreAccidentValidationDataset
from utils.accident_training_dataset import PreAccidentTrainingDataset
from utils.augmented_dataset import AugmentedVideoDataset  # Import the new augmented dataset class
import logging
from typing import Tuple, Dict, Union, List

import torch.nn as nn
from utils.tool import AverageMeter, Monitor
from utils.misc import print_trainable_parameters
import time
import os
from experiment.pseudo_Dataset import PseduoDataset
import math
import numpy as np
import torch
from utils.tool import get_device, set_seed
import argparse
from torchvision import transforms
import torch.nn.functional as F
from datetime import datetime
from models.model import DSA_RNN
from models.model import baseline_model
from utils.YamlArguments import load_yaml_file_from_arg
from utils.CommandLineArguments import train_parse_args

from models.model import get_model
from utils.optim import get_optimizer
from utils.loss import AnticipationLoss, TemporalBinaryCrossEntropy
from torch.amp import autocast, GradScaler
from utils.stats import case_counting
from sklearn.metrics import average_precision_score

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




def get_dataloaders(args, logger) -> Tuple[torch.utils.data.DataLoader, Union[torch.utils.data.DataLoader, None]]:
    '''
    Create and return data loaders for training and validation.

    Args:
        args: Command line arguments
        logger: Logger object
        val_ratio: Ratio of validation data (if not using separate validation set)

    Returns:
        train_dataloader(torch.utils.data.DataLoader)
        val_dataloader(torch.utils.data.DataLoader)
    '''
    # Configure augmentations based on augmentation_types argument
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
    if aug_config['horizontal_flip']:
        logger.info(f"Horizontal flip probability: {args.horizontal_flip_prob}")

    # Use the standard dataset if augmentation is disabled
    logger.info("Using standard PreAccidentTrainDataset without augmentation")
    train_dataset = PreAccidentTrainingDataset(
        root_dir = args.training_dir,
        csv_file= args.training_csv,
        num_frames = 16,
        frame_window = 16,
        interested_interval = 100,
        resize_shape = (128, 171),
        crop_size = (112, 112),
        augmentation_config=aug_config,
        global_augment_prob=args.augmentation_prob,
        horizontal_flip_prob=args.horizontal_flip_prob,
    )

    # Handle debug mode
    if args.debug:
        max_size = 64
        indices = list(range(len(train_dataset)))
        indices = indices[:max_size]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    # Create data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    if args.debug:
        return train_loader, None
    
    val_dataset = PreAccidentTrainingDataset(
        root_dir = args.validation_dir,
        csv_file = args.validation_csv,
        num_frames = 16,
        frame_window = 16,
        interested_interval = 100,
        resize_shape = (128, 171),
        crop_size = (112, 112),
        global_augment_prob = 0.0,
        horizontal_flip_prob = 0.0,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    return train_loader, val_loader

def get_eval_dataloaders(args, logger) -> Dict[str, torch.utils.data.DataLoader]:
    val_loaders = dict()
    
    logger.info('Loading the evaluation dataset with multiple pre-accident time intervals (500ms, 1000ms, and 1500ms).')
    for pre_accident_time in ['500', '1000', '1500']:
        val_dataset = PreAccidentValidationDataset(
            root_dir = f'{args.evaluation_dir}/tta_{pre_accident_time}ms',
            csv_file = args.evaluation_csv 
        )
        # For validation, always use the standard dataset (no augmentation)
    
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        val_loaders[pre_accident_time] =  val_loader
    return val_loaders
    


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
    train_loss_over_iterations = []

    max_iter = EPOCHS * len(train_loader)
    dataset_per_epoch = len(train_loader)
    start = time.time()

    logger.info(f'Train-procedure with {len(train_loader)} iterations')
    for mini_batch_index, data in enumerate(train_loader):
        data_time.update(time.time() - start)
        X, target, T_diff = data
        X = X.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        with autocast(device_type = device.type):
            output = model(X)
            loss = criterion(output, target, T_diff)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10) # Gradient clip
        
        SCALER.scale(loss).backward()
        SCALER.step(optimizer)
        SCALER.update()
        

        # Positive and Negative cases counting
        positive, negative, true_case, false_case = case_counting(mode = 'volume', output = output, target = target)
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
        train_loss_over_iterations.append(loss_meter.current_value)

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
        if DEBUG:
            print(f'[DEBUGMODE] Main Loop broken due to args.debug')
            break


    return {'mPrec': (TP_meter.sum)/(TP_meter.sum + FP_meter.sum  + EPS),
            'mRecall': TP_meter.sum/(TP_meter.sum + FN_meter.sum + EPS),
            'mLoss': loss_meter.avg_value,
            'mAcc': (TP_meter.sum + TN_meter.sum)/(true_meter.sum + false_meter.sum + EPS),
            'Loss_record': train_loss_over_iterations
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
            X, target, T_diff = data
            X = X.to(device)
            target = target.to(device)
            T_diff = T_diff.to(device)
            with autocast(device_type = device.type):
                output = model(X)
                loss = criterion(output, target, T_diff)
            
            # Positive and Negative cases counting
            positive, negative, true_case, false_case = case_counting(mode = 'volume', output = output, target = target)

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

            if (((mini_batch_index + 1) % PRINT_FREQ) == 0) or (mini_batch_index + 1 == len(val_loader)):
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
    #logger.info(f'Epoch[{epoch + 1:03d}/{EPOCHS:03d}] mean Average Precision on 500ms/1000ms/1500ms: {mAP/3:.3f}')

    return {'mPrec': (TP_meter.sum)/(TP_meter.sum + FP_meter.sum  + EPS),
            'mRecall': TP_meter.sum/(TP_meter.sum + FN_meter.sum + EPS),
            'mLoss': loss_meter.avg_value,
            'mAcc': (TP_meter.sum + TN_meter.sum)/(true_meter.sum + false_meter.sum + EPS),
                }

def mAP_evaluation(val_loaders: Dict[str, torch.utils.data.DataLoader], model: torch.nn.Module) -> float:
    
    model.eval()
    logger.info('===> Processing mean averaged precision (mAP) with multiple pre-accident time intervals.')
    mAP = 0
    with torch.no_grad():
        for pre_accident_time in ['500', '1000', '1500']:
            logger.info(f'Processing with pre-accident time intervals: {pre_accident_time} ms...')
            outputs = []
            targets = []
            for data in val_loaders[pre_accident_time]:
                X, y = data
                X = X.to(device)
                y = y.to(device)
                with autocast(device_type = device.type):
                    output = model(X)
                prob = F.softmax(output, dim = 1)
                outputs += prob[:, 1].tolist()
                targets += y.tolist()
            outputs = torch.tensor(outputs).to('cpu')
            targets = torch.tensor(targets).to('cpu')
            mAP += average_precision_score(targets, outputs)
    mAP = float(mAP/3)
    logger.info(f'===> Current mean averaged precision: {mAP}')
    return mAP
     


def main():

    global logger, device, EPOCHS, PRINT_FREQ, DEBUG, LR_RATE, BATCH_SIZE, EPS, NUM_WORKERS, RESIZE_SHAPE, SCALER, STATS_MODE
    import sys
    use_yaml_file = len(sys.argv) == 1+1 and '.yaml' in sys.argv[1]

    args = load_yaml_file_from_arg(sys.argv[1]) if use_yaml_file else train_parse_args()


    print(f"Training with batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Number of epochs: {args.epochs}")
    if args.augmentation_types:
        print(f"Using augmentation types: {args.augmentation_types}")
        print(f"Global augmentation probability: {args.augmentation_prob}")

    BATCH_SIZE = args.batch_size
    PRINT_FREQ = args.print_freq
    EPOCHS= args.epochs
    LR_RATE = args.learning_rate
    DEBUG = args.debug # debug mode if --debug is added
    EPS = 1e-8 # small number to avoid zero-division
    NUM_WORKERS = args.num_workers
    RESIZE_SHAPE = (112, 112) if args.model_type == 'baseline' else (224, 224) # used to set up the resize shape of frame
    #STATS_MODE = 'volume' if args.model_type in ['baseline', 'timesformer', 'swintransformer'] else 'sequence'

    set_seed(123)
    logger = get_logger()
    device = get_device()

    logger.info(f'Set-up model: {args.model_type}')
    logger.info("=> creating model")

    model = get_model(model_type = args.model_type)()
    model.to(device)

    np = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total number of parameters in model: {np}")
    logger.info("Trainable Architecture Components:")
    print_trainable_parameters(model, logger = logger)
    Loss_fn = TemporalBinaryCrossEntropy(decay_coefficient = args.decay_coefficient) 
    #AnticipationLoss(decay_nframe = DECAY_NFRAME, pivot_frame_index = 100, device = get_device())
    logger.info("=> Creating optimizer")
    logger.info(f"Set up optimizer: {args.optimizer}")
    optimizer = get_optimizer(args.optimizer)([p for p in model.parameters() if p.requires_grad], lr = LR_RATE*0.1 if args.optimizer.lower=='lion' else LR_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6)
    SCALER = GradScaler()
    
    logger.info(f'{optimizer}')
    logger.info(f'Total number of epochs: {EPOCHS}')
    logger.info(f'Load dataset...')

    # Updated call to get_dataloaders to pass args
    train_dataloader, val_dataloader = get_dataloaders(args, logger)
    eval_dataloaders = get_eval_dataloaders(args = args, logger = logger)

    os.makedirs(args.model_dir, exist_ok = True) # save model parameters under this folder
    os.makedirs(args.monitor_dir, exist_ok = True) # save training details under this folder

    # Generate a tag that describes the configuration
    if args.augmentation_types:
        # Create a shorter tag for augmentation types
        aug_types_str = '_'.join([t[:3] for t in sorted(args.augmentation_types)])
        aug_tag = f'_aug{args.augmentation_prob:.2f}_{aug_types_str}'
    else:
        aug_tag = ''

    tag = f'bs{BATCH_SIZE}_lr{LR_RATE}{aug_tag}'

    # Log the tag being used
    logger.info(f"Using tag: {tag}")

    iterations_per_epoch = len(train_dataloader.dataset) // train_dataloader.batch_size + int(len(train_dataloader.dataset) % train_dataloader.batch_size != 0)
    monitor = Monitor(save_path = args.monitor_dir, tag = tag, iterations_per_epoch = iterations_per_epoch)


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
            mAP = mAP_evaluation(val_loaders = eval_dataloaders, model = model)
            valid_metrics['mAP'] = mAP 
            
            scheduler.step(valid_metrics['mLoss'])
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f'Current learning rate: {current_lr:.8f}')

            if prev_loss > valid_metrics['mLoss']:
                torch.save(model.state_dict(), f'{args.model_dir}/best_model_ckpt_{tag}.pt')
                torch.save(optimizer.state_dict(), f'{args.model_dir}/best_optim_ckpt_{tag}.pt')
                best_point_metrics.update(valid_metrics)
                best_point_metrics['current_epoch'] = epoch + 1
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
