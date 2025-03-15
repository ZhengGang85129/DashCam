import torch
import os
import torch
from torch.utils.data import Dataset
import cv2
from typing import List, Tuple, Union
import pandas as pd
from torchvision import transforms
import numpy as np
import math


import torch
import random
import torch.nn.functional as F
from torchvision.transforms import functional as TF


class RandomShortestSize:
    """
    Resize video frames so that the shortest side is randomly chosen from a range of sizes
    while maintaining the aspect ratio.
    
    Args:
        min_size (int): Minimum size for the shortest side
        max_size (int): Maximum size for the shortest side
        interpolation (str): Interpolation method ('bilinear', 'nearest', 'bicubic')
    """
    def __init__(self, min_size, max_size, interpolation='bilinear'):
        self.min_size = min_size
        self.max_size = max_size
        self.interpolation = interpolation
        
    def __call__(self, video):
        """
        Args:
            video (Tensor): Video tensor of shape (T, C, H, W) or (C, T, H, W)
                            where T is number of frames, C is channels
        
        Returns:
            Tensor: Resized video tensor with same shape format
        """
        # Determine the input format
        if video.shape[1] == 3:  # (T, C, H, W) format
            T, C, H, W = video.shape
            channel_dim = 1
        else:  # (C, T, H, W) format
            C, T, H, W = video.shape
            channel_dim = 0
        
        # Randomly choose a size for the shortest side
        shortest_side = random.randint(self.min_size, self.max_size)
        
        # Calculate new dimensions maintaining aspect ratio
        if H < W:
            new_H = shortest_side
            new_W = int(W * (new_H / H))
        else:
            new_W = shortest_side
            new_H = int(H * (new_W / W))
        
        # Resize each frame
        if channel_dim == 1:  # (T, C, H, W) format
            resized_video = torch.zeros((T, C, new_H, new_W), dtype=video.dtype, device=video.device)
            for t in range(T):
                frame = video[t].unsqueeze(0)  # Add batch dimension for F.interpolate
                resized_frame = F.interpolate(
                    frame, 
                    size=(new_H, new_W), 
                    mode=self.interpolation,
                    align_corners=False if self.interpolation != 'nearest' else None
                )
                resized_video[t] = resized_frame.squeeze(0)
                
        else:  # (C, T, H, W) format
            # For this format, we can reshape to (C, T*1, H, W) and use interpolate once
            reshaped = video.reshape(C, T*1, H, W)
            resized = F.interpolate(
                reshaped,
                size=(new_H, new_W),
                mode=self.interpolation,
                align_corners=False if self.interpolation != 'nearest' else None
            )
            resized_video = resized.reshape(C, T, new_H, new_W)
        
        return resized_video
    
    def __repr__(self):
        return f"{self.__class__.__name__}(min_size={self.min_size}, max_size={self.max_size}, interpolation='{self.interpolation}')"
    
class MultipleRandomCrop(object):
    def __init__(self, crop_size:Tuple = (224, 224), num_crops:int = 5):
        self.crop_size = crop_size
        self.num_crops = num_crops
    def __call__(self, video: torch.Tensor):
        '''
        Args: 
            video (n_frames, channels, Height, Width)
        Return: (multiple random crops of same clip)
            crops (n_crops, n_frames, channels, Height, Width)
        '''
        C, T, H, W = video.shape
        crop_H, crop_W = self.crop_size
        
        crops = torch.zeros((self.num_crops, T, C, crop_H, crop_W))
        
        for i in range(self.num_crops):
            top = random.randint(0, H - crop_H)
            right = random.randint(0, W - crop_W)
            for t in range(T):
                crops[i, t] = video[:, t, top:top+crop_H, right:right+crop_W]
        return crops


class VideoDataset(Dataset):
    """
    Custom dataset for loading video files from a directory.
    """
    def __init__(
        self,
        root_dir: str = "./dataset/train/extracted",
        csv_file: str = './dataset/train.csv',
        #transform: Optional[transforms.Compose] = None,
    ):
        """
        Initialize the video dataset.
        Args:
            root_dir (str): Directory containing the video files
            transform (transforms.Compose, optional): Transform to be applied to frames
            resize_shape (tuple): Target size for frame resizing (height, width)
        """
        self.root_dir = root_dir
        self.data_frame = pd.read_csv(csv_file)
        self.video_indices = self.data_frame['id'].to_list() 
        # Get all MP4 files in the directory
        self.video_files = dict()
        global_index = 0
        for Index in self.video_indices:
            file = os.path.join(root_dir, f'{Index:05d}.mp4')
            if os.path.isfile(file):
                self.video_files[global_index] = (file, self.data_frame[self.data_frame['id'] == Index]['target'].item())
                global_index += 1 
                
        if not self.video_files:
            raise RuntimeError(f"No MP4 files found in {root_dir}")

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
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize frame
            #frame = cv2.resize(frame, (560, 560))
            # Apply transforms
            frames.append(torch.from_numpy(frame.astype(np.float32)) / 255.0)
            frame_count += 1
        assert frame_count == 100  
        cap.release()
        
        # Ensure we have enough frames
        return frames 
        

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a video clip from the dataset.
        
        Args:
            idx (int): Index of video file
            
        Returns:
            torch.Tensor: Tensor of shape (n_frames, channels, height, width)
        """
        video_path,  target = self.video_files[idx]
        frames = self.__load_video(video_path)
        # Stack frames into a single tensor
        clip = torch.stack(frames)
        return clip.permute(0, 3, 1, 2), target


class VideoTo3DImageDataset(Dataset):
    """
    Custom dataset for loading video files from a directory.
    """
    def __init__(
        self,
        root_dir: str = "./dataset/train",
        csv_file: str = './dataset/train.csv',
        num_frames: int = 16,
        mode: str = 'train',
        strategy: str = 'default'
        #transform: Optional[transforms.Compose] = None,
    ):
        """
        Initialize the video dataset.
        Args:
            root_dir (str): Directory containing the video files
            transform (transforms.Compose, optional): Transform to be applied to frames
            resize_shape (tuple): Target size for frame resizing (height, width)
        """
        self.root_dir = root_dir
        self.data_frame = pd.read_csv(csv_file)
        self.video_indices = self.data_frame['id'].to_list() 
        # Get all MP4 files in the directory
        self.video_files = dict()
        global_index = 0
        self.num_frames = num_frames 
        self.mode = mode
        
        if strategy == 'default':
            self.transforms = transforms.Compose([
                transforms.ToTensor(),  
                transforms.Resize(size = (128, 171)),
                transforms.Normalize(mean = [0.43216, 0.394666, 0.37645], std = [0.22803, 0.22145, 0.216989]),
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),  
            ])
        
        if self.mode == 'train':
            if strategy == 'default':
                self.transforms_train = transforms.Compose([
                    transforms.RandomCrop((112, 112)), #Resize the 720 x 1280 -> 112 x 112
                ]) 
            else:    
                self.transforms_train = transforms.Compose([
                    RandomShortestSize(min_size = 128, max_size = 160),
                    transforms.RandomCrop((112, 112)), #Resize the 720 x 1280 -> 112 x 112
                ]) 
        elif self.mode == 'validation' or self.mode == 'inference':
            if strategy == 'default':
                self.transforms_eval = transforms.Compose([
                    transforms.CenterCrop((112, 112))
                ])
            else:
                self.transforms_eval = transforms.Compose([
                    MultipleRandomCrop(crop_size = (112, 112), num_crops = 3),
                ])
        else:
            raise ValueError(f'Mode: {self.mode} does not exist.')
         
        for Index in self.video_indices:
            file = os.path.join(root_dir, f'{Index:05d}.mp4')
            if os.path.isfile(file):
                if self.mode == 'train' or self.mode == 'validation':
                    self.video_files[global_index] = (file, self.data_frame[self.data_frame['id'] == Index]['target'].item())
                else:
                    self.video_files[global_index] = (file, Index)
                global_index += 1 
                
        if not self.video_files:
            raise RuntimeError(f"No MP4 files found in {root_dir}")

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
        
        for i in range(0, total_frames, interval):
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, frame = video.read()
            if not success:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(self.transforms(frame.astype(np.float32) / 255.0))
            frames_saved += 1
            if frames_saved >= self.num_frames:
                break
                
        video.release()
        return frames 

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[torch.Tensor, int]]:
        """
        Get a video clip from the dataset.
        
        Args:
            idx (int): Index of video file
            
        Returns:
            torch.Tensor: Tensor of shape (n_frames, channels, height, width)
        """
        # Stack frames into a single tensor
        if self.mode == 'train':
            video_path,  target = self.video_files[idx]
            frames = self.__load_video(video_path)
            frames = torch.stack(frames, dim = 1).permute(1, 0, 2, 3)
            frames = self.transforms_train(frames)
            return frames , target
        elif self.mode == 'validation':
            video_path, target = self.video_files[idx]
            frames = self.__load_video(video_path)
            frames = torch.stack(frames, 1).permute(1, 0, 2, 3)
            frames = self.transforms_eval(frames) # (Batch size, n_crops, n_Frames, n_channels, Height, Width)
            return frames, target
        else:
            video_path, id = self.video_files[idx]
            frames = self.__load_video(video_path)
            frames = torch.stack(frames, 1).permute(1, 0, 2, 3)
            frames = self.transforms_eval(frames) # (Batch size, n_crops, n_Frames, n_channels, Height, Width)
            return frames, id
            

# Example usage:
if __name__ == "__main__":
    # Create dataset
    dataset = VideoTo3DImageDataset(
        root_dir="./dataset/train",
        mode = 'validation'
    )
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=10,
        shuffle = False,
        num_workers=4,
        pin_memory = True
    )
    
    # Example iteration
    for batch in dataloader:
        # batch shape: (batch_size, clip_len, channels, height, width)
        video, target = batch
        print(video.shape)
        break