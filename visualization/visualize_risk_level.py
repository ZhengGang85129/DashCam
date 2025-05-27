import torch
import cv2
from typing import Tuple
from utils.tool import get_device
import matplotlib.pyplot as plt
import pandas as pd 
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch.nn.functional as F
from models.model import load_model
from src.datasets.transforms import CustomTransforms

class _Config:
    nframes: int = 16
    model_type:str = 'mvit_v2'
    plot_region: int = 200
    input_folder: Path = Path('./dataset/img_database/validation')
    output_folder: Path = Path('visualization/output')
    metadata: Path = Path('dataset/img_database/frame-metadata_validation.csv')
    weights: Path = Path('weights/full_model-mvit2_best_v0.pt')
    img_size: Tuple[int, int] = (224, 224)
    resize_shape: Tuple[int, int] = (128, 171)
    stride : int = 1
    verbose: int = 0
    
class Predictor:
    def __init__(self, args, manager):
        self.metadata = pd.read_csv(_Config.metadata)
        self.nframes = _Config.nframes
        self.plot_region =  _Config.plot_region
        self.transforms = CustomTransforms(model_type = _Config.model_type)
        
        self.model = load_model(args, manager)
        
    def load_video_frame(self, video_id: int = 343) -> None:
        
        frame_metadata = self.metadata[self.metadata.video_id == video_id]
        self.plot_region = min(self.plot_region, frame_metadata.last_frame.unique().item() - self.nframes + 1) 
        frames_in_window = frame_metadata[(frame_metadata.last_frame - self.plot_region - self.nframes  + 1 < frame_metadata.frame) &(frame_metadata.frame <= frame_metadata.last_frame)] 
        
        img_folder = _Config.input_folder / Path(f'video_{video_id:05d}')
        
        self.raw_images = [ cv2.cvtColor(cv2.imread(str(img_folder / Path(f'{frame:05d}.jpg'))), cv2.COLOR_BGR2RGB) for frame in frames_in_window.frame.to_list()]


    def get_data(self) -> torch.Tensor:
        
        frame_asbatch = []
        frame_remain = self.plot_region
        start_idx = 0
        while frame_remain:
            frames = []
            if _Config.verbose:
                print(f"Start to read frame[{start_idx}:{start_idx+_Config.nframes}]") 
            frames = self.transforms.get_transforms(self.raw_images[start_idx:start_idx+_Config.nframes], mode = 'inference')
            assert len(frames) == _Config.nframes
            frame_asbatch.append(torch.stack(frames))
            start_idx += 1
            frame_remain -= 1 
        self.input_data = torch.stack(frame_asbatch)
        if _Config.verbose:
            print(self.input_data.shape) 
        return self.input_data
     
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
    import os
    from utils.YamlArguments import load_yaml_file_from_arg
    from utils.strategy_manager import get_strategy_manager
    config_path = os.getenv("CONFIG_YAML", "configs/mvit2.yaml")
    video_id = os.getenv("VIDEO_ID", "366")
    use_yaml_file = config_path
    args = load_yaml_file_from_arg(use_yaml_file)
    args.device = get_device() 
    if args is None:
        raise ValueError('Please provide configuration file.')
    manager = get_strategy_manager(args.training_strategy) 
    _Config.output_folder.mkdir(parents = True, exist_ok=True)
    device = get_device()
    
    predictor = Predictor(args, manager)

    predictor.load_video_frame(video_id = video_id)
    input = predictor.get_data()
    
    predictor.visualize()
