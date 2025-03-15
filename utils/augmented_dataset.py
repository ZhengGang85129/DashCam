import torch
from torch.utils.data import Dataset
import cv2
import pandas as pd
from torchvision import transforms
import numpy as np
import math
import os
import random
from typing import List, Tuple

class AugmentedVideoDataset(Dataset):
    """
    Enhanced dataset for loading video files with data augmentation techniques.
    """
    def __init__(
        self,
        root_dir: str = "./dataset/train",
        csv_file: str = './dataset/train.csv',
        num_frames: int = 16,
        augment: bool = True,
        use_advanced_transforms: bool = True,
    ):
        """
        Initialize the augmented video dataset.
        Args:
            root_dir (str): Directory containing the video files
            csv_file (str): Path to CSV file with video metadata
            num_frames (int): Number of frames to sample from each video
            augment (bool): Whether to apply data augmentation
            use_advanced_transforms (bool): Whether to apply advanced augmentation techniques
        """
        self.root_dir = root_dir
        self.data_frame = pd.read_csv(csv_file)
        self.video_indices = self.data_frame['id'].to_list() 
        self.video_files = dict()
        global_index = 0
        self.num_frames = num_frames 
        self.augment = augment
        self.use_advanced_transforms = use_advanced_transforms
        
        # Basic transforms that are always applied
        self.base_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((112, 112)),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], 
                                std=[0.22803, 0.22145, 0.216989])
        ])
        
        # Basic augmentation transforms
        self.aug_transforms = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
            ], p=0.5),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
            ], p=0.3),
            transforms.RandomHorizontalFlip(p=0.5),
            # Simulating fog/noise
            transforms.RandomApply([
                transforms.Lambda(lambda x: self._add_noise(x, noise_factor=0.02))
            ], p=0.3),
            transforms.RandomApply([
                transforms.Lambda(lambda x: self._add_fog(x, fog_intensity=0.15))
            ], p=0.3),
        ])
        
        # Advanced transforms (optional)
        self.advanced_transforms = get_advanced_transforms(apply_prob=0.3)
         
        for Index in self.video_indices:
            file = os.path.join(root_dir, f'{Index:05d}.mp4')
            if os.path.isfile(file):
                self.video_files[global_index] = (file, self.data_frame[self.data_frame['id'] == Index]['target'].item())
                global_index += 1 
                
        if not self.video_files:
            raise RuntimeError(f"No MP4 files found in {root_dir}")

    def _add_noise(self, img, noise_factor=0.1):
        """Add random noise to an image tensor"""
        noise = torch.randn_like(img) * noise_factor
        noisy_img = img + noise
        return torch.clamp(noisy_img, 0., 1.)
    
    def _add_fog(self, img, fog_intensity=0.3):
        """Simulate fog effect by adding a bright overlay with reduced contrast"""
        fog = torch.ones_like(img) * fog_intensity
        foggy_img = img * (1 - fog_intensity) + fog
        return torch.clamp(foggy_img, 0., 1.)

    def __len__(self) -> int:
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
            
        # Calculate interval
        interval = math.floor(total_frames / self.num_frames)

        frames_saved = 0
        frames = []
        
        # Determine if we'll add rain effect to the entire video
        apply_rain_to_video = self.augment and self.use_advanced_transforms and random.random() < 0.2
        
        for i in range(0, total_frames, interval):
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, frame = video.read()
            if not success:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to tensor first using base transforms
            frame_tensor = self.base_transforms(frame.astype(np.float32) / 255.0)
            
            # Apply basic augmentations during training
            if self.augment:
                frame_tensor = self.aug_transforms(frame_tensor)
                
                # Apply advanced transforms if enabled
                if self.use_advanced_transforms:
                    frame_tensor = self.advanced_transforms(frame_tensor)
                    
                # Apply rain effect consistently across all frames in the video if selected
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
        Get a video clip from the dataset.
        
        Args:
            idx (int): Index of video file
            
        Returns:
            torch.Tensor: Tensor of shape (channels, n_frames, height, width)
            int: Target label
        """
        video_path, target = self.video_files[idx]
        frames = self.__load_video(video_path)
        # Stack frames into a single tensor
        
        return torch.stack(frames, dim=1), target


# Additional transforms that can be used independently
def get_advanced_transforms(apply_prob=0.5):
    """
    Create more sophisticated augmentation transforms.
    
    Args:
        apply_prob (float): Probability of applying each augmentation
        
    Returns:
        transforms.Compose: Composed transforms
    """
    return transforms.Compose([
        # Spatial augmentations
        transforms.RandomApply([
            transforms.RandomRotation(15)
        ], p=apply_prob),
        transforms.RandomApply([
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
        ], p=apply_prob),
        
        # Color/intensity augmentations
        transforms.RandomApply([
            transforms.RandomGrayscale(p=0.2)
        ], p=apply_prob),
        
        # Add more sophisticated weather simulations
        transforms.RandomApply([
            transforms.Lambda(lambda x: simulate_rain(x, drop_length=20, drop_width=1, drop_count=20))
        ], p=apply_prob * 0.5),
    ])

def simulate_rain(img, drop_length=20, drop_width=1, drop_count=20):
    """
    Simulate rain by adding random streaks.
    This is a simple approximation for demonstration.
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
