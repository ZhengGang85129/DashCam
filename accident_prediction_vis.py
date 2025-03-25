import torch
import cv2
from typing import Tuple
from torchvision import transforms
import os 
import math
from models.baseline_model import baseline_model
from utils.tool import get_device
import torch.nn as nn
from typing import Optional
import matplotlib.pyplot as plt
import torch.nn.functional as F
from matplotlib.gridspec import GridSpec
import sys
from utils.misc import parse_args
import argparse
from models.model import get_model
'''
[usage] python3 ./accident_prediction_vis.py --clip_path dataset/train/train_video/00043.mp4 --model_ckpt CHECKPOINT --filename probability --model_type [baseline:timesformer:swintransformer]
'''

def argument_parser() -> argparse.ArgumentParser:
    parser = parse_args('prob_vis') 
    parser.add_argument('--clip_path', type = str, help = 'the extracted clip path.')
    parser.add_argument('--model_ckpt', type = str, help = 'check point path to the model')
    parser.add_argument('--model_type', type = str)
    parser.add_argument('--filename', type = str, default = 'probability')
    _args = parser.parse_args()
    return _args
def denormalize(tensor, mean = [0.43216, 0.394666, 0.37645], std = [0.22803, 0.22145, 0.216989]):
    """
    Denormalizes a tensor that was normalized with mean and std.
    
    Args:
        tensor (torch.Tensor): Normalized image tensor [C, H, W]
        mean (list): Mean values used for normalization
        std (list): Std values used for normalization
        
    Returns:
        torch.Tensor: Denormalized image tensor
    """
    # Create tensors of mean and std
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    # Denormalize: multiply by std and add mean (inverse of normalization)
    return tensor * std + mean

def load_model(state_path: Optional[str] = None) -> nn.Module:
    model = get_model(model_type = args.model_type)() 
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


def frame_swin(video_path: str, 
                num_frames: int,
                frame_window: int, 
                interested_interval: int = 100,
                resize_shape: Tuple[int, int] = (128, 171),
                crop_size: Tuple[int, int] = (112, 112),
                normalize: bool = True):
    
    T = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(resize_shape),
            transforms.CenterCrop(
                crop_size
            ),
            transforms.Normalize(
                    mean=[0.43216, 0.394666, 0.37645], 
                    std=[0.22803, 0.22145, 0.216989]),
        ]
    )
    
    interval = math.floor(frame_window/num_frames)
    print(interval) 
    window_stacks = []
    
    for selected_frame in range(15, interested_interval + 15):
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        start_frame = selected_frame - interval * num_frames  + 1
        
        frames_saved = 0
        frames = []
        
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        #print('start: ', start_frame, 'selected:', selected_frame)
        for Idx, frame_index in enumerate(range(start_frame, selected_frame + interval, interval)):
            success, frame = video.read()
            if not success:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#shape: (720, 1280, 3)
            frame = T(frame) #shape: (3, 112, 112)
            frames_saved += 1
            frames.append(frame)
        if frames_saved < num_frames:
            raise RuntimeError(f'total saved frames: {frames_saved}. start frame: {start_frame}, end frame: {selected_frame}, total_frame: {total_frames}')
        window_stacks.append(torch.stack(frames, dim = 1).permute(1, 0, 2, 3))       
    video.release()
    #return frames, selected_frame

    return torch.stack(window_stacks, dim = 0)  
        


if __name__ == "__main__":
    #Idx = int(sys.argv[1]) 
    global device, args
    device = get_device()
    args = parse_args()
    print('Video loading')
    print('Create model')
    model = load_model(args.model_ckpt)
    model.eval()
    
    video_path = args.clip_path 
    window_stacks = frame_swin(video_path = video_path, 
                num_frames = 16,
                frame_window = 16, 
                interested_interval = 100,
                resize_shape= (128, 171),
                crop_size= (112, 112)) # Apply sliding-window approach to extract sequential frame sequences from the input video.
    
    print('Sliding Window size: ', window_stacks.shape) #96 x 16 x 3 x 112 x 112

    print('Inferencing...')
    probs = []
    T_diffs  = torch.tensor([90 - i for i in range(90)] + [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10])
    for i in range(10):
        output = model(window_stacks[i* 10: (i + 1) * 10])
        target = torch.ones((10)).to(torch.long)
        prob = F.softmax(output, dim = 1)
        probs += prob[:, 1].tolist()
    
    frame_indices = [i for i in range(100)]
    
    fig = plt.figure(figsize=(15, 8))
    gs = GridSpec(2, 10, height_ratios=[3, 1])
    
    for i in range(10):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(denormalize(window_stacks[i* 10, 15]).permute(1, 2, 0))
        ax.axis('off')
    ax_prob = fig.add_subplot(gs[1, :])
    ax_prob.plot(frame_indices, prob, 'r--')
    ax_prob.set_xlabel('Frame Index')
    ax_prob.set_ylabel('Probability')
    ax_prob.set_ylim(0, 1)
    ax_prob.axhline(y= 0.5, color = 'pink', linestyle = '-', alpha = 0.6, label = f'Threshold')
    for frame_index in [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]:
        ax_prob.axvline(x = frame_index, color='blue', linestyle='--', alpha=0.7, 
                   label=f'Frame {frame_index}' if frame_index == 15 else "") 
    ax_prob.axvline(x = 74, color='green', linestyle='--', alpha=0.7, label=f'500 ms before accident') 
    ax_prob.axvline(x = 59, color='green', linestyle='--', alpha=0.7, label=f'1000 ms before accident') 
    ax_prob.axvline(x = 44, color='green', linestyle='--', alpha=0.7, label=f'1500 ms before accident') 
    
    plt.legend() 
    plt.tight_layout()
    os.makedirs('visualization', exist_ok = True)
    plt.savefig(f'visualization/{args.filename}.png')
    plt.savefig(f'visualization/{args.filename}.pdf')
    print(f'Check=> visualization/{args.filename}.pdf')
    print(f'Check=> visualization/{args.filename}.png')