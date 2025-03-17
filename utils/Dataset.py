import torch
import os
import torch
from torch.utils.data import Dataset
import cv2
from typing import List, Tuple
import pandas as pd
from torchvision import transforms
import numpy as np
import math


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
        resize_shape: Tuple[int, int] = (112, 112)
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
        self.transforms = transforms.Compose([
            transforms.ToTensor(),  # Convert (H, W, C) to (C, H, W)
            transforms.Resize(resize_shape), #Resize the 720 x 1280 -> 112 x 112
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
        ])
         
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
            frames.append( self.transforms(frame.astype(np.float32) / 255.0))
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
            torch.Tensor: Tensor of shape (n_frames, channels, height, width)
        """
        video_path,  target = self.video_files[idx]
        frames = self.__load_video(video_path)
        # Stack frames into a single tensor
        
        return torch.stack(frames, dim=1) , target

class VideoTo3DImageDataset_Inference(Dataset):
    """
    Custom dataset for loading video files from a directory.
    """
    def __init__(
        self,
        root_dir: str = "./dataset/test",
        csv_file: str = './dataset/test.csv',
        num_frames: int = 16,
        resize_shape: Tuple[int, int] = (112, 112)
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
        self.transforms = transforms.Compose([
            transforms.ToTensor(),  # Convert (H, W, C) to (C, H, W)
            transforms.Resize(resize_shape), #Resize the 720 x 1280 -> 112 x 112
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
        ])
         
        for Index in self.video_indices:
            file = os.path.join(root_dir, f'{Index:05d}.mp4')
            if os.path.isfile(file):
                self.video_files[global_index] = (file,  Index)
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
            frames.append( self.transforms(frame.astype(np.float32) / 255.0))
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
            torch.Tensor: Tensor of shape (n_frames, channels, height, width)
        """
        video_path,  Index = self.video_files[idx]
        frames = self.__load_video(video_path)
        # Stack frames into a single tensor
        
        return torch.stack(frames, dim=1) , Index
# Example usage:
if __name__ == "__main__":
    # Create dataset
    dataset = VideoTo3DImageDataset(
        root_dir="./dataset/train/extracted",
    )
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle = False,
        num_workers=4,
        pin_memory = True
    )
    
    # Example iteration
    for batch in dataloader:
        # batch shape: (batch_size, clip_len, channels, height, width)
        video, target = batch
        print(video.shape, target.shape)