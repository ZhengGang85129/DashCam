from torchvision import transforms
from typing import Tuple, List
import torchvision.transforms.functional as F 
import torch
from functools import partial


class CustomTransforms:
    def __init__(self, model_type: str ):
        self.model_type = model_type
        
        assert self.model_type in ['swintransformer', 'baseline', 'mvit_v2']
    
    def get_transforms(self, frames: List, mode: str)-> List[torch.Tensor]:
        
        resize_shape = (112, 112) if self.model_type == 'baseline' else (224, 224)
        #transformed_frames = []
        _transform = []
        _transform.append(transforms.ToTensor())
        if mode == 'training':
            angle = float(torch.rand(1) * 20 - 10)
            scale_factor = float(torch.rand(1) * (1 - resize_shape[0]/256) + resize_shape[0]/256)
            new_size = int(256 * scale_factor), int(256 * scale_factor)
            
            max_x = new_size[0] - resize_shape[0]
            max_y = new_size[1] - resize_shape[1]
            
            i = int(torch.rand(1) * max_y)
            j = int(torch.rand(1) * max_x)
            h, w = resize_shape
            
            do_hflip = torch.rand(1) < 0.5
            
            color_jitter = transforms.ColorJitter(brightness=0.3, contrast = 0.3, saturation = 0.3)
            
            color_jitter_params = [
                param[0].item() if isinstance(param, torch.Tensor) else param
                for param in color_jitter.get_params(
                    color_jitter.brightness,
                    color_jitter.contrast,
                    color_jitter.saturation,
                    color_jitter.hue
                )
            ]
            
            _transform.append(partial(F.rotate, angle = angle))
            
            _transform.append(partial(F.resize, size = new_size, antialias=True))
            
            _transform.append(partial(F.crop, top = i, left = j, height = h, width = w))
            
            if do_hflip:
                _transform.append(F.hflip)
            
            _transform.append(partial(F.adjust_brightness, brightness_factor = color_jitter_params[0])) 
            _transform.append(partial(F.adjust_contrast, contrast_factor = color_jitter_params[1]))
            _transform.append(partial(F.adjust_saturation, saturation_factor = color_jitter_params[2]))
        
        else:
            _transform.append(partial(F.resize, size = resize_shape, antialias=True))

        if self.model_type == 'baseline':
            _transform.append(transforms.Normalize(
                mean=[0.43216, 0.394666, 0.37645], 
                std=[0.22803, 0.22145, 0.216989]))
        else:    
            _transform.append(transforms.Normalize(
                mean = [0.45, 0.45, 0.45], 
                std = [0.225, 0.225, 0.225]))
        new_frames = []
        for frame in frames: 
            
            new_frame = frame
            for T in _transform:
                new_frame = T(new_frame)
            new_frames.append(new_frame)
        return new_frames
if __name__ == "__main__":
    transform = CustomTransforms(model_type = 'baseline') 
    pass          