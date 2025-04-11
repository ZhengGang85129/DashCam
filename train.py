from utils.tool import get_device, set_seed
import torch
from utils.accident_validation_dataset import PreAccidentValidationDataset
from utils.accident_training_dataset import PreAccidentTrainingDataset
import logging
from typing import Dict, Tuple

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



from models.model import get_model
from utils.optim import get_optimizer
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
    formatted_time = current_time.strftime("%m%d%H%M")
    logging.basicConfig(
        filename = f'training-lr{LR_RATE}-bs{BATCH_SIZE}-{formatted_time}.log',
        level=logging.INFO,
    )

    return logger




def get_dataloader(args, logger) -> Tuple[torch.utils.data.DataLoader]:
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
        sampling_approach = args.sampling_approach
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
        batch_size= strategy_manager.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_dataset = PreAccidentTrainingDataset(
        root_dir = args.validation_dir,
        csv_file= args.validation_csv,
        num_frames = 16,
        frame_window = 16,
        interested_interval = 100,
        resize_shape = (128, 171),
        crop_size = (112, 112),
        sampling_approach = args.sampling_approach
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size= strategy_manager.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    return train_loader, val_loader

def get_eval_dataloaders(args, logger) -> Dict[str, Dict[str, torch.utils.data.DataLoader]]:
    eval_dataloaders = {
        'train': dict(),
        'val': dict()
    }
    
    logger.info('Loading the evaluation dataset with multiple pre-accident time intervals (500ms, 1000ms, and 1500ms).')
    for pre_accident_time in ['500', '1000', '1500']:
        val_dataset = PreAccidentValidationDataset(
            root_dir = f'{args.evaluation_dir}/tta_{pre_accident_time}ms',
            csv_file = args.evaluation_csv 
        )
        # For validation, always use the standard dataset (no augmentation)
    
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=strategy_manager.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        eval_dataloaders['val'][pre_accident_time] =  val_loader
        
        train_dataset = PreAccidentValidationDataset(
            root_dir = f'{args.evaluation_train_dir}/tta_{pre_accident_time}ms',
            csv_file = args.evaluation_train_csv 
        )
        # For validation, always use the standard dataset (no augmentation)
    
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=strategy_manager.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        eval_dataloaders['train'][pre_accident_time] =  train_loader
    return eval_dataloaders
    


def train(train_loader: torch.utils.data.DataLoader, model: torch.nn.Module, criterion: torch.nn.Module, epoch: int, optimizer: torch.optim.Optimizer) -> float:

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    # Metric relavant meter
    loss_meter = AverageMeter()
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
        train_loss_over_iterations.append(loss_meter.current_value)

        if (((mini_batch_index + 1) % PRINT_FREQ) == 0) or (mini_batch_index + 1 == len(train_loader)):
            logger.info(f'Epoch: [{epoch + 1:03d}/{EPOCHS:03d}][{mini_batch_index + 1:03d}/{dataset_per_epoch}] '
                        f'Data {data_time.current_value:.1f} s ({data_time.avg_value:.1f} s) '
                        f'Batch {batch_time.current_value:.1f} s ({batch_time.avg_value:.1f} s) '
                        f'Remain {remain_time} '
                        f'Loss[temporal-weighted] {(loss_meter.current_value):.3f} ({(loss_meter.avg_value):.3f}) '
                        )
        start = time.time()
        if DEBUG:
            print(f'[DEBUGMODE] Main Loop broken due to args.debug')
            break

    return train_loss_over_iterations, loss_meter.avg_value

def validate(val_loader: torch.utils.data.DataLoader, model: torch.nn.Module, criterion: torch.nn.Module, epoch: int) -> float:

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    # Metric relavant meter
    loss_meter = AverageMeter()

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
            with autocast(device_type = device.type):
                output = model(X)
                loss = criterion(output, target, T_diff)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10) # Gradient clip
            

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

            if (mini_batch_index + 1 == len(val_loader)):
                logger.info(f'Epoch: [{epoch + 1:03d}/{EPOCHS:03d}][{mini_batch_index + 1:03d}/{dataset_per_epoch}] '
                            f'Data {data_time.current_value:.1f} s ({data_time.avg_value:.1f} s) '
                            f'Batch {batch_time.current_value:.1f} s ({batch_time.avg_value:.1f} s) '
                            f'Remain {remain_time} '
                            f'Loss[temporal-weighted] {(loss_meter.current_value):.3f} ({(loss_meter.avg_value):.3f}) '
                            )
            start = time.time()
            if DEBUG:
                print(f'[DEBUGMODE] Main Loop broken due to args.debug')
                break

    return loss_meter.avg_value

def mAP_evaluation(val_loaders: Dict[str, torch.utils.data.DataLoader], model: torch.nn.Module, criterion: nn.Module, epoch: int = 0) -> float:
    
    model.eval()
    logger.info('===> Processing mean averaged precision (mAP) with multiple pre-accident time intervals.')
    mAP = 0
    TP_meter = AverageMeter() #True positive
    FP_meter = AverageMeter() #False postive
    FN_meter = AverageMeter() #False negative
    TN_meter = AverageMeter() #True negative
    loss_meter = AverageMeter()
    true_meter = AverageMeter()
    false_meter = AverageMeter()
    with torch.no_grad():
        for pre_accident_time in ['500', '1000', '1500']:
            logger.info(f'Processing with pre-accident time intervals: {pre_accident_time} ms...')
            outputs = []
            targets = []
            batch_time = AverageMeter()
            data_time = AverageMeter()
            start = time.time()
            dataset_per_epoch = len(val_loaders[pre_accident_time])
            max_iter = EPOCHS * dataset_per_epoch
            
            
            for mini_batch_index, data in enumerate(val_loaders[pre_accident_time]):
                data_time.update(time.time() - start)
                X, y = data
                X = X.to(device)
                y = y.to(device)
                with autocast(device_type = device.type):
                    output = model(X)
                prob = F.softmax(output, dim = 1)
                loss = criterion(output, y, torch.zeros_like(y))
                batch_time.update(time.time() - start) 
                positive, negative, true_case, false_case = case_counting(mode = 'volume', output = output, target = y)
                
                current_iter = epoch * dataset_per_epoch + mini_batch_index + 1
                remain_iter = max_iter - current_iter
                remain_time = batch_time.avg_value * remain_iter
                t_m, t_s = divmod(remain_time, 60)
                t_h, t_m = divmod(t_m, 60)
                remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
                loss_meter.update(loss.item())
                true_meter.update(true_case.sum().item())
                false_meter.update(false_case.sum().item() )
                TP_meter.update(torch.logical_and(positive, true_case).sum().item())
                FP_meter.update(torch.logical_and(positive, false_case).sum().item())
                TN_meter.update(torch.logical_and(negative, false_case).sum().item())
                FN_meter.update(torch.logical_and(negative, true_case).sum().item())
                
                outputs += prob[:, 1].tolist()
                targets += y.tolist()
                if (mini_batch_index % 10 == 1) or (mini_batch_index == dataset_per_epoch - 1):
                    logger.info(
                    f'Epoch: [{epoch + 1:03d}/{EPOCHS:03d}][{mini_batch_index + 1:03d}/{dataset_per_epoch}] '
                    f'Data {data_time.current_value:.1f} s ({data_time.avg_value:.1f} s) '
                    f'Batch {batch_time.current_value:.1f} s ({batch_time.avg_value:.1f} s) '
                    f'Remain {remain_time} ')
                start = time.time()
            outputs = torch.tensor(outputs).to('cpu')
            targets = torch.tensor(targets).to('cpu')
            mAP += average_precision_score(targets, outputs)
    mAP = float(mAP/3)
    #logger.info(f'===> Loss/Mean Averaged Precision/Precision/Recall/Accuracy: {}{mAP:.3f}/{(TP_meter.sum)/(TP_meter.sum + FP_meter.sum  + EPS):.3f}/{}')
    logger.info(
        f'mLoss: {loss_meter.avg_value:.3f} '
        f'mAP: {mAP:.3f} '
        f'mAcc: {(TP_meter.sum + TN_meter.sum)/(true_meter.sum + false_meter.sum + EPS):.3f} '
        f'mRecall: {TP_meter.sum/(TP_meter.sum + FN_meter.sum + EPS):.3f} '
        f'mPrec: {(TP_meter.sum)/(TP_meter.sum + FP_meter.sum  + EPS):.3f} '
        f'mSpec: {(TN_meter.sum)/(TN_meter.sum + FP_meter.sum + EPS):.3f}'
    )
    
    return {'mPrec': (TP_meter.sum)/(TP_meter.sum + FP_meter.sum  + EPS),
            'mRecall': TP_meter.sum/(TP_meter.sum + FN_meter.sum + EPS),
            'mLoss': loss_meter.avg_value,
            'mAcc': (TP_meter.sum + TN_meter.sum)/(true_meter.sum + false_meter.sum + EPS),
            'mSpec': (TN_meter.sum)/(TN_meter.sum + FP_meter.sum + EPS),
            'mAP': mAP
                }
     


def main():

    set_seed(123)
    global logger, device, EPOCHS, PRINT_FREQ, DEBUG, LR_RATE, BATCH_SIZE, EPS, NUM_WORKERS, RESIZE_SHAPE, SCALER, STATS_MODE, strategy_manager
    import sys
    use_yaml_file = len(sys.argv) == 1+1 and '.yaml' in sys.argv[1]

    args = load_yaml_file_from_arg(sys.argv[1]) if use_yaml_file else train_parse_args()
    strategy_manager = get_strategy_manager(args.strategy)

    print(f"Training with batch size: {strategy_manager.batch_size}")
    print(f"Learning rate: {strategy_manager.optimizer['lr']}")
    print(f"Number of epochs: {strategy_manager.epochs}")
    if args.augmentation_types:
        print(f"Using augmentation types: {args.augmentation_types}")
        print(f"Global augmentation probability: {args.augmentation_prob}")

    BATCH_SIZE = strategy_manager.batch_size
    PRINT_FREQ = args.print_freq
    EPOCHS= strategy_manager.epochs
    LR_RATE = strategy_manager.optimizer['lr']
    DEBUG = args.debug # debug mode if --debug is added
    EPS = 1e-8 # small number to avoid zero-division
    NUM_WORKERS = args.num_workers
    RESIZE_SHAPE = (224, 224) if args.model_type == 'swintransformer' else (112, 112) # used to set up the resize shape of frame
    #STATS_MODE = 'volume' if args.model_type in ['baseline', 'timesformer', 'swintransformer'] else 'sequence'

    logger = get_logger()
    device = get_device()

    logger.info(f'Set-up model: {args.model_type}')
    logger.info("=> creating model")

    model = get_model(model_type = args.model_type)(trainable_parts = strategy_manager.trainable_parts, classifier = args.classifier)
    
    if strategy_manager.resume:
        logger.info(f'Model loads saved state from: {strategy_manager.check_point_path}')
        model.load_state_dict(torch.load(strategy_manager.check_point_path))
            

        #model.load(strategy_manager.check_point_path)
    
    model.to(device)

    np = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total number of trainable parameters in model: {np}")
    logger.info("Trainable Architecture Components:")
    print_trainable_parameters(model, logger = logger)
    Loss_fn = TemporalBinaryCrossEntropy(decay_coefficient = args.decay_coefficient) 
    #AnticipationLoss(decay_nframe = DECAY_NFRAME, pivot_frame_index = 100, device = get_device())
    logger.info(f"=> Creating optimizer: {strategy_manager.optimizer['name']}")
    
    
    optimizer = get_optimizer(params = model.named_parameters(), optimizer = strategy_manager.optimizer)
    if strategy_manager.optim_resume and strategy_manager.resume:
        logger.info(f'Optimizer loads saved state from: {strategy_manager.optim_check_point_path}')
        optimizer.load_state_dict(torch.load(strategy_manager.optim_check_point_path))
        
    logger.info(f"=> Creating Scheduler: {strategy_manager.scheduler['name']}")
    scheduler = get_scheduler(scheduler = strategy_manager.scheduler, optimizer = optimizer) 
    SCALER = GradScaler()
    
    logger.info(f'{optimizer}')
    logger.info(f'Total number of epochs: {EPOCHS}')
    logger.info(f'Load dataset...')

    # Updated call to get_dataloaders to pass args
    train_dataloader, val_loader = get_dataloader(args, logger)
    eval_dataloaders = get_eval_dataloaders(args = args, logger = logger)

    os.makedirs(strategy_manager.model_dir, exist_ok = True) # save model parameters under this folder
    os.makedirs(strategy_manager.monitor_dir, exist_ok = True) # save training details under this folder

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
    monitor = Monitor(save_path = strategy_manager.monitor_dir, tag = tag, iterations_per_epoch = iterations_per_epoch)


    best_point_metrics = {
        'mLoss': float('inf'),
        'mPrec': 0,
        'mRecall': 0,
        'mAcc': 0,
        'mAP': 0,
        'mSpec': 0, 
        'current_epoch': 0
    }

    prev_loss = math.inf

    for epoch in range(EPOCHS):
        logger.info('Training...')
        LossRecord, meanLossRecord = train(train_loader = train_dataloader, model = model, epoch = epoch, optimizer = optimizer, criterion = Loss_fn)
        logger.info('Evaluating...')
        
        meanLossRecord_val = validate(val_loader = val_loader, model = model, epoch = epoch, criterion = Loss_fn)
        
        if not DEBUG:
            logger.info('==== Training sample ====')
            train_metrics = mAP_evaluation(val_loaders = eval_dataloaders['train'], model = model, criterion = Loss_fn, epoch = epoch)
            logger.info('==== Validation sample ====')
            valid_metrics = mAP_evaluation(val_loaders = eval_dataloaders['val'], model = model, criterion = Loss_fn, epoch = epoch)
            
            train_metrics['Loss_record'] = LossRecord 
            train_metrics['meanLossRecord'] = meanLossRecord 
            valid_metrics['meanLossRecord'] = meanLossRecord_val
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f'Current learning rate: {current_lr:.8f}')

            if prev_loss > valid_metrics['mLoss']:
                torch.save(model.state_dict(), f'{strategy_manager.model_dir}/best_model_ckpt_{tag}.pt')
                torch.save(optimizer.state_dict(), f'{strategy_manager.model_dir}/best_optim_ckpt_{tag}.pt')
                best_point_metrics.update(valid_metrics)
                best_point_metrics['current_epoch'] = epoch + 1
                prev_loss = valid_metrics['mLoss']

            monitor.update(metrics = {
                'train': train_metrics,
                'validation': valid_metrics,
                'best_point': best_point_metrics
            })

        torch.save(model.state_dict(), f'{strategy_manager.model_dir}/model_ckpt-epoch{epoch:02d}_{tag}.pt')
        torch.save(optimizer.state_dict(), f'{strategy_manager.model_dir}/optim_ckpt-epoch{epoch:02d}_{tag}.pt')
    return

if __name__ == "__main__":
    main()
