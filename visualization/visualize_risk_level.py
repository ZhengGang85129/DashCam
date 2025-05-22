import torch
import cv2
from typing import Tuple
from torchvision import transforms
from utils.tool import get_device
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from utils.misc import parse_args
from models.model import get_model
import pandas as pd 
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import OrderedDict
import torch.nn.functional as F

class _Config:
    nframes: int = 16
    plot_region: int = 200
    input_folder: Path = Path('./dataset/img_database/train')
    output_folder: Path = Path('visualization/output')
    metadata: Path = Path('dataset/img_database/frame-metadata_train.csv')
    img_size: Tuple[int, int] = (112, 112)
    resize_shape: Tuple[int, int] = (128, 171)
    stride : int = 1
    verbose: int = 0
    
class AccidentDataset:
    def __init__(self, metadata: str):
        self.metadata = pd.read_csv(_Config.metadata)
        self.nframes = _Config.nframes
        self.plot_region =  _Config.plot_region
    
        
    def load_video_frame(self, video_id: int = 343) -> None:
        
        frame_metadata = self.metadata[self.metadata.video_id == video_id]
        
        frames_in_window = frame_metadata[(frame_metadata.last_frame - self.plot_region - self.nframes  + 1 < frame_metadata.frame) &(frame_metadata.frame <= frame_metadata.last_frame)] 
        
        img_folder = _Config.input_folder / Path(f'video_{video_id:05d}')
        
        self.raw_images = [ cv2.cvtColor(cv2.imread(str(img_folder / Path(f'{frame:05d}.jpg'))), cv2.COLOR_BGR2RGB) for frame in frames_in_window.frame.to_list()]


    @property
    def transforms(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.ToTensor(), # Convert (H, W, C) with range [0, 255] into (C, H, W) with range (0, 1)
                transforms.Resize(_Config.resize_shape, antialias=True), #ex: Resize the 720 x 1280 ->  resize_shape (128 x 171 by default)
                transforms.CenterCrop(_Config.img_size),# crop the center of images with size: crop_size(112 x 112 by default)
                transforms.Normalize(
                    mean=[0.43216, 0.394666, 0.37645], 
                    std=[0.22803, 0.22145, 0.216989]), # normalize the values.
            ]
        )
    def get_data(self) -> torch.Tensor:
        
        frame_asbatch = []
        frame_remain = self.plot_region
        start_idx = 0
        while frame_remain:
            frames = []
            if _Config.verbose:
                print(f"Start to read frame[{start_idx}:{start_idx+_Config.nframes}]") 
            for idx in range(_Config.nframes):
                frame_idx = start_idx + idx
                
                frame =self.raw_images[frame_idx]
                #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.transforms(frame)
                frames.append(frame)
            assert len(frames) == _Config.nframes
            frame_asbatch.append(torch.stack(frames))
            start_idx += 1
            frame_remain -= 1 
        self.input_data = torch.stack(frame_asbatch)
        if _Config.verbose:
            print(self.input_data.shape) 
        return self.input_data
    
    def create_model(self, model_type: str, state_path: str, device: torch.cuda.device) -> None:
        self.model = get_model(model_type = model_type)()
        saved_state = torch.load(state_path, map_location = device if device is not None else 'cpu')
        
        state_dict = saved_state['model'] if 'model' in saved_state else saved_state
    
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'fc' in k:
                new_key = k.replace('model.fc', 'classifier')
            else: 
                new_key = k.replace('model.', 'backbone.')
            new_state_dict[new_key] = v

        self.model.load_state_dict(new_state_dict)  
        self.model.eval()
        
     
    def visualize(self, figsize: Tuple[int, int] = (8, 6), fps = 30) -> None:    
       
        fig, (ax_img, ax_plot) = plt.subplots(2, 1, figsize = figsize, gridspec_kw={'height_ratios': [4, 1]}) 
        ax_img.axis('off')
        ax_plot.set_xlim(0, _Config.plot_region - 1)
        ax_plot.set_ylim(0, 1)
        ax_plot.set_xlabel("Frames(FPS = 30)", fontsize = 14, fontweight = 'bold', labelpad = 10)
        ax_plot.set_ylabel("Risk Level",fontsize = 14, fontweight = 'bold', labelpad = 10)
        ax_plot.set_facecolor("#f9f9f9")
        ax_plot.grid(True, linestyle = '--', alpha = 0.3)
        ax_plot.axhline(y = 0.5, color = 'blue', linestyle = '-.', alpha = 0.6, label = 'Threshold') 
        ims = []
        probs = []
        for idx, img in enumerate(self.raw_images[_Config.nframes-1:]):
            im = ax_img.imshow(img, animated = True) 
             
            with torch.no_grad():
                prob = F.softmax(self.model(self.input_data[idx].unsqueeze(0)), dim = -1).tolist()[0][1]
                probs.append(prob)
            line, = ax_plot.plot(range(len(probs)), probs, color = '#007ACC', marker = 'o', markersize = 5, linewidth = 2, alpha = 0.8)
            marker = ax_plot.scatter([idx], [prob], color = '#FF6347', s = 50, zorder = 3)
            ims.append([im, line, marker])
            
        ani = animation.ArtistAnimation(fig, ims, interval = 1000//fps, blit = True,repeat = True)

        ani.save(Path(f'vid.gif'), writer = 'pillow', dpi = 80)
    


if __name__ == '__main__':
    import sys
    _Config.output_folder.mkdir(parents = True, exist_ok=True)
    video_id = int(sys.argv[1]) 
    device = get_device()
     
    dataset = AccidentDataset(metadata = _Config.metadata)

    dataset.load_video_frame(video_id = video_id)
    input = dataset.get_data()
    
    dataset.create_model(model_type = 'baseline', state_path = 'baseline_data_aug.pt', device = device)
    dataset.visualize()
