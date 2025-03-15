import torch
from torch.utils.data import Dataset
import cv2
import pandas as pd
from torchvision import transforms
import numpy as np
import math
import os
import random
from typing import List, Tuple, Dict, Optional, Union

class AugmentedVideoDataset(Dataset):
    """
    Enhanced dataset for loading video files with configurable data augmentation techniques.
    
    This dataset allows for fine-grained control over different augmentation methods,
    with individual toggles and probability settings for each augmentation type.
    
    Supported augmentations:
    - Fog effect: Simulates foggy conditions by adding a bright overlay
    - Noise: Adds random noise to simulate poor camera conditions
    - Gaussian blur: Blurs the image to simulate focus issues
    - Color jitter: Adjusts brightness, contrast, saturation, and hue
    - Horizontal flip: Mirrors the image horizontally
    - Rain effect: Simulates rain drops in video frames
    """
    def __init__(
        self,
        root_dir: str = "./dataset/train",
        csv_file: str = './dataset/train.csv',
        num_frames: int = 16,
        augmentation_config: Optional[Dict[str, bool]] = None,
        augmentation_prob: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the augmented video dataset.
        
        Args:
            root_dir (str): Directory containing the video files
            csv_file (str): Path to CSV file with video metadata
            num_frames (int): Number of frames to sample from each video
            augmentation_config (dict): Booleans to enable specific augmentations
                Supported keys: 'fog', 'noise', 'gaussian_blur', 'color_jitter', 
                                'horizontal_flip', 'rain_effect'
            augmentation_prob (dict): Probabilities to apply each augmentation
                Values should be between 0.0 and 1.0
        """
        # Default configuration (no augmentation)
        self.aug_config = {
            'fog': False,
            'noise': False,
            'gaussian_blur': False,
            'color_jitter': False,
            'horizontal_flip': False,
            'rain_effect': False,
        }

        # Default probabilities
        self.aug_prob = {
            'fog': 0.3,
            'noise': 0.3,
            'gaussian_blur': 0.3,
            'color_jitter': 0.3,
            'horizontal_flip': 0.3,
            'rain_effect': 0.3,
        }

        # Override with user configuration if provided
        if augmentation_config:
            self._validate_config(augmentation_config)
            for key, value in augmentation_config.items():
                if key in self.aug_config:
                    self.aug_config[key] = value

        if augmentation_prob:
            self._validate_probabilities(augmentation_prob)
            for key, value in augmentation_prob.items():
                if key in self.aug_prob:
                    self.aug_prob[key] = value
        
        self.root_dir = root_dir
        self.data_frame = pd.read_csv(csv_file)
        self.video_indices = self.data_frame['id'].to_list() 
        self.video_files = dict()
        global_index = 0
        self.num_frames = num_frames 
        
        # Basic transforms that are always applied
        self.base_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((112, 112)),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], 
                                std=[0.22803, 0.22145, 0.216989])
        ])
        
        # Individual augmentation transforms
        self.apply_fog = transforms.RandomApply(
            [transforms.Lambda(lambda x: self._add_fog(x, fog_intensity=0.15))], 
            p=self.aug_prob['fog']
        )
        self.apply_noise = transforms.RandomApply(
            [transforms.Lambda(lambda x: self._add_noise(x, noise_factor=0.02))], 
            p=self.aug_prob['noise']
        )
        self.apply_gaussian_blur = transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], 
            p=self.aug_prob['gaussian_blur']
        )
        self.apply_color_jitter = transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)], 
            p=self.aug_prob['color_jitter']
        )
        self.apply_horizontal_flip = transforms.RandomHorizontalFlip(
            p=self.aug_prob['horizontal_flip']
        )
        
        # Load videos from directory
        for Index in self.video_indices:
            file = os.path.join(root_dir, f'{Index:05d}.mp4')
            if os.path.isfile(file):
                self.video_files[global_index] = (
                    file, 
                    self.data_frame[self.data_frame['id'] == Index]['target'].item()
                )
                global_index += 1 
                
        if not self.video_files:
            raise RuntimeError(f"No MP4 files found in {root_dir}")

    def _validate_config(self, config: Dict[str, bool]) -> None:
        """
        Validate the augmentation configuration dictionary.
        
        Args:
            config (Dict[str, bool]): Configuration dictionary to validate
        
        Raises:
            ValueError: If any keys are unknown or values are not boolean
        """
        valid_keys = set(self.aug_config.keys())
        for key, value in config.items():
            if key not in valid_keys:
                raise ValueError(f"Unknown augmentation type: {key}. "
                                f"Valid options are: {', '.join(valid_keys)}")
            if not isinstance(value, bool):
                raise ValueError(f"Augmentation config values must be boolean, got {type(value)} for {key}")

    def _validate_probabilities(self, probs: Dict[str, float]) -> None:
        """
        Validate the augmentation probability dictionary.
        
        Args:
            probs (Dict[str, float]): Probability dictionary to validate
        
        Raises:
            ValueError: If any keys are unknown or values are not between 0 and 1
        """
        valid_keys = set(self.aug_prob.keys())
        for key, value in probs.items():
            if key not in valid_keys:
                raise ValueError(f"Unknown augmentation type: {key}. "
                                f"Valid options are: {', '.join(valid_keys)}")
            if not isinstance(value, (float, int)):
                raise ValueError(f"Probability values must be numeric, got {type(value)} for {key}")
            if value < 0 or value > 1:
                raise ValueError(f"Probability values must be between 0 and 1, got {value} for {key}")

    def _add_noise(self, img: torch.Tensor, noise_factor: float = 0.1) -> torch.Tensor:
        """
        Add random noise to an image tensor.
        
        Args:
            img (torch.Tensor): Input image tensor of shape [C, H, W]
            noise_factor (float): Intensity of the noise to add
        
        Returns:
            torch.Tensor: Noisy image tensor
        """
        noise = torch.randn_like(img) * noise_factor
        noisy_img = img + noise
        return torch.clamp(noisy_img, 0., 1.)
    
    def _add_fog(self, img: torch.Tensor, fog_intensity: float = 0.3) -> torch.Tensor:
        """
        Simulate fog effect by adding a bright overlay with reduced contrast.
        
        Args:
            img (torch.Tensor): Input image tensor of shape [C, H, W]
            fog_intensity (float): Intensity of the fog effect (0-1)
        
        Returns:
            torch.Tensor: Image tensor with fog effect
        """
        fog = torch.ones_like(img) * fog_intensity
        foggy_img = img * (1 - fog_intensity) + fog
        return torch.clamp(foggy_img, 0., 1.)

    def __len__(self) -> int:
        """
        Return the number of videos in the dataset.
        
        Returns:
            int: Dataset length
        """
        return len(self.video_files)

    def __load_video(self, video_path: str) -> List[torch.Tensor]:
        """
        Load video file and return preprocessed frames.
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            List[torch.Tensor]: List of preprocessed frames
        """
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            
        # Calculate interval for frame sampling
        interval = math.floor(total_frames / self.num_frames)

        frames_saved = 0
        frames = []
        
        # Determine if we'll use rain effects to the entire video
        apply_rain_to_video = self.aug_config['rain_effect'] and random.random() < self.aug_prob['rain_effect']
        
        for i in range(0, total_frames, interval):
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, frame = video.read()
            if not success:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to tensor first using base transforms
            frame_tensor = self.base_transforms(frame.astype(np.float32) / 255.0)
            
            # Apply augmentations according to configuration
            if self.aug_config['fog']:
                frame_tensor = self.apply_fog(frame_tensor)
            if self.aug_config['noise']:
                frame_tensor = self.apply_noise(frame_tensor)
            if self.aug_config['gaussian_blur']:
                frame_tensor = self.apply_gaussian_blur(frame_tensor)
            if self.aug_config['color_jitter']:
                frame_tensor = self.apply_color_jitter(frame_tensor)
            if self.aug_config['horizontal_flip']:
                frame_tensor = self.apply_horizontal_flip(frame_tensor)
            if apply_rain_to_video:
                frame_tensor = simulate_rain(
                    frame_tensor, 
                    drop_length=random.randint(1, 5),
                    drop_count=random.randint(10, 40)
                )
                
            frames.append(frame_tensor)
            frames_saved += 1
            if frames_saved >= self.num_frames:
                break
                
        video.release()
        return frames 

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a video clip and its label from the dataset.
        
        Args:
            idx (int): Index of video file
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Video tensor of shape (channels, n_frames, height, width)
                - Target label tensor
        """
        video_path, target = self.video_files[idx]
        frames = self.__load_video(video_path)
        # Stack frames into a single tensor
        return torch.stack(frames, dim=1), target

def simulate_rain(img: torch.Tensor, drop_length: int = 20, drop_width: int = 1, drop_count: int = 20) -> torch.Tensor:
    """
    Simulate rain by adding random streaks to an image.
    
    Args:
        img (torch.Tensor): Input image tensor of shape [C, H, W]
        drop_length (int): Length of rain drops
        drop_width (int): Width of rain drops
        drop_count (int): Number of rain drops to add
    
    Returns:
        torch.Tensor: Image tensor with rain effect
    """
    c, h, w = img.shape
    rain_img = img.clone()
    
    for _ in range(drop_count):
        x = random.randint(0, w-1)
        y = random.randint(0, h-drop_length-1)
        
        # Add a white streak
        rain_value = torch.ones(c, drop_length, drop_width) * 0.8
        
        # Ensure we don't go out of bounds
        end_y = min(y + drop_length, h)
        end_x = min(x + drop_width, w)
        
        # Add the rain drop
        rain_img[:, y:end_y, x:end_x] = rain_img[:, y:end_y, x:end_x] * 0.2 + rain_value[:, :end_y-y, :end_x-x]
    
    return torch.clamp(rain_img, 0., 1.)
