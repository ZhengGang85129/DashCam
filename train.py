import torch
import torch.optim.optimizer
import torch.utils.data.dataloader
from traindataset_frames_extraction import get_video_info
from Dataset import VideoDataset # type: ignore
import logging
from typing import List, Tuple
from loss import AnticipationLoss
from model import DSA_RNN


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

def train(train_loader: torch.utils.data.DataLoader, model: torch.nn.Module, criterion: torch.nn.Module, epoch: int, optimizer: torch.optim.optimizer) -> Tuple[float, float]:
    
    mPrec, mRecall = 0.0, 0.0
    for mini_batch_index, data in enumerate(train_loader):
        X, target = data
        output = model(X)
        
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    
    return (mPrec, mRecall) 

def evaluation()->Tuple[float, float]:
    mPrec, mRecall = 0.0, 0.0
    return (mPrec, mRecall) 

def main():
    global logger, device
    logger = get_logger()
    device = get_device()
    
    model =  DSA_RNN(hidden_size = 256)
    Loss_fn = AnticipationLoss()
    optimizer = torch.optim.RAdam(model.parameters(), 
                                  betas=(0.95, 0.999), 
                                  lr = 0.001)

    model.to(device)
    epochs = 10
    
    train_dataloader = get_dataloader() 
     
    for epoch in range(epochs):
        train(train_loader = train_dataloader, model = model, epoch = epoch, optimizer = optimizer, criterion = Loss_fn)
        #FIX ME evaluation(...) 
    return



if __name__ == "__main__":
    
    main()    





