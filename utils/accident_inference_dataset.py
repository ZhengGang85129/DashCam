import torch
from torch.utils.data import Dataset
import cv2
from typing import List, Tuple
from torchvision import transforms
import pandas as pd
import os
import math
import numpy as np
 
class PreAccidentInferenceDataset(Dataset):
    '''
    The sampling approach is the sliding window.
    '''
    def __init__(
        self,
        root_dir: str = 'dataset/test',
        csv_file: str = './dataset/test.csv',
        num_frames: int = 16,
        frame_window: int = 16,
        resize_shape: Tuple[int, int] = (128, 171),
        crop_size: Tuple[int, int] = (112, 112)
    ):
        """
        Args:
        root_dir (str):
        csv_file (str):
        num_sample_frames:
        resize_shape: 
        """
        self.root_dir = root_dir 
        self.data_frame = pd.read_csv(csv_file)
        self.video_indices = self.data_frame['id'].to_list()
        # Get all MP4 files in the directory
        self.video_files = dict()
        
        global_index = 0
        
        self.num_frames = num_frames 
        self.frame_window = frame_window
        self.transforms1 = transforms.Compose([
            transforms.ToTensor(), # Convert (H, W, C) with range [0, 255] into (C, H, W) with range (0, 1)
            transforms.Resize(resize_shape),
            #ex: Resize the 720 x 1280 ->  resize_shape
            transforms.CenterCrop(crop_size)
        ])
         
        self.transforms2 = transforms.Compose([
            transforms.Normalize(
                mean=[0.43216, 0.394666, 0.37645], 
                std=[0.22803, 0.22145, 0.216989]),
        ])
        
        for Index in self.video_indices:
            file = os.path.join(
                root_dir, f'{Index:05d}.mp4'
            )
            if os.path.isfile(file):
                self.video_files[global_index] = (file, Index)
                global_index += 1 
        if not self.video_files:
            raise RuntimeError(f"No MP4 files found in {root_dir}")
        
    def __len__(self) -> int:
        return len(self.video_files)
    
    def __load_video(self, video_path: str) -> List[torch.Tensor]:
        """
        Load video file and return preprocessed frames.
        """
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
         
        # Calculate interval
        interval = math.floor(self.frame_window / self.num_frames)
        end_frame = total_frames - 1
        start_frame = end_frame - (interval) * (self.num_frames - 1)
        frames_saved = 0
        frames = []
        
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for Idx, frame_index in enumerate(range(start_frame, end_frame + 1, interval)):
            success, frame = video.read()
            if not success:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#shape: (720, 1280, 3)
            frame = self.transforms1(frame) #shape: (3, 112, 112)
            frame = self.transforms2(frame) #shape: (3, 112, 112)
            frames_saved += 1
            frames.append(frame)
        if frames_saved < self.num_frames:
            raise RuntimeError()
                
        video.release()
        return frames 
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            idx: index for the video.
        Reture:
            frames(batch_size, num_frames, n_channels, height, width): Tensor containing video frames with dimensions representing batch samples, temporal sequence, color channels, and spatial resolution. 
            Idx(batch_size): Video Id.
        """ 
        video_path, Idx = self.video_files[idx]
        frames = self.__load_video(video_path)
        frames = torch.stack(frames, dim = 1).permute(1, 0, 2, 3) 
        return frames, Idx


if __name__ == '__main__':
    tta_dataset = PreAccidentInferenceDataset(
        root_dir = 'dataset/train/validation_video/tta_500ms/',
        csv_file = 'dataset/validation_videos.csv',
        num_sample_frames = 16,
        frame_window = 16,
        resize_shape = (128, 171),
        crop_size = (112, 112)
        )
    
    data_loader = torch.utils.data.DataLoader(
        tta_dataset,
        batch_size = 16,
        shuffle=False,
        num_workers = 8,
        pin_memory=True
    )
        
    for idx, (X, y) in enumerate(data_loader):
        print(X.shape)    
         
