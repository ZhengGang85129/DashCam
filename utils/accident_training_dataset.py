import torch
from torch.utils.data import Dataset
import cv2
from typing import List, Tuple
from torchvision import transforms
import pandas as pd
import os 
import math
import numpy as np
import random

class PreAccidentTrainingDataset(Dataset):
    '''
    The sampling approach is the sliding window.
    '''
    def __init__(
        self,
        root_dir: str = 'dataset/train/train_video',
        csv_file: str = './dataset/extracted_train.csv',
        num_frames: int = 16,
        frame_window: int = 16,
        interested_interval: int = 100,
        resize_shape: Tuple[int, int] = (128, 171),
        crop_size: Tuple[int, int] = (112, 112)
    ):
        self.root_dir = root_dir
        self.data_frame = pd.read_csv(csv_file)
        
        self.video_indices = self.data_frame['id'].to_list()
        self.video_files = dict()
        
        global_index = 0
        self.num_frames = num_frames
        self.frame_window = frame_window
        self.interested_interval = interested_interval 
        
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(), # Convert (H, W, C) with range [0, 255] into (C, H, W) with range (0, 1)
                transforms.Resize(resize_shape), #ex: Resize the 720 x 1280 ->  resize_shape (128 x 171 by default)
                transforms.CenterCrop(crop_size),# crop the center of images with size: crop_size(112 x 112 by default)
                transforms.Normalize(
                    mean=[0.43216, 0.394666, 0.37645], 
                    std=[0.22803, 0.22145, 0.216989]), # normalize the values.
                
            ]
        )
        
        
        for Index in self.video_indices:
            file = os.path.join(
                root_dir, f'{Index:05d}.mp4'
            )
            if os.path.isfile(file):
                start_frame = self.data_frame[self.data_frame['id'] == Index]['start_frame'].item()
                end_frame = self.data_frame[self.data_frame['id'] == Index]['end_frame'].item()
                if end_frame - start_frame + 1 < self.interested_interval + self.frame_window - 1:
                    continue  
                self.video_files[global_index] = (file, self.data_frame[self.data_frame['id'] == Index]['target'].item(), 
                self.data_frame[self.data_frame['id'] == Index]['start_frame'].item(), 
                self.data_frame[self.data_frame['id'] == Index]['end_frame'].item(), 
                self.data_frame[self.data_frame['id'] == Index]['event_frame'].item())
                if 
                global_index += 1 
        if not self.video_files:
            raise RuntimeError(f"No MP4 files found in {root_dir}")
    def __len__(self) -> int:
        return len(self.video_files)
    
    def __load_video(self, video_path: str, event_frame: int) -> Tuple[List[torch.Tensor], int]:
        
        # Calculate interval
        interval = math.floor(self.frame_window / self.num_frames) 
        
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
         
        
        selected_frame = random.randint(0, self.interested_interval - 1) + (self.num_frames - 1) * interval # Randomly select a frame as the 'current frame'. 
        start_frame = selected_frame - (self.num_frames  - 1)* interval  # Indices falling within the range [start_frame, current_frame] represent past frames.
        frames_saved = 0
        frames = []
        
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        count = 0
        illed_frame = 0
        for Idx, frame_index in enumerate(range(start_frame , selected_frame + interval, interval)):
            success, frame = video.read()
            if not success:
                #print(frame_index)
                illed_frame = frame_index
                break
            count += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#shape: (720, 1280, 3)
            frame = self.transforms(frame) #shape: (3, 112, 112)
            frames_saved += 1
            frames.append(frame)
            if count > self.num_frames:
                raise ValueError
        
        if frames_saved < self.num_frames:
            print('not success')
            print(f'video_path: {video_path}')
            print(f'total saved frames: {frames_saved}. start frame: {selected_frame - (self.num_frames - 1) * interval}, end_frame: {selected_frame}, total_frame: {total_frames}, illed_frame: {illed_frame}')
            #raise RuntimeError(f'total saved frames: {frames_saved}. start frame: {selected_frame - (self.num_frames - 1) * interval}, end_frame: {selected_frame}, total_frame: {total_frames}, illed_frame: {illed_frame}')
                
        video.release()
        return frames, event_frame - selected_frame
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Args:
            idx: index for the video.
        Reture:
            frames(batch_size, num_frames, n_channels, height, width): Tensor containing video frames with dimensions representing batch samples, temporal sequence, color channels, and spatial resolution.
            target(batch_size): whether the frames are from the video that contains the accident.
            T_diff(batchsize): difference of selected frame to accident frame.
        """ 
        video_path, target, _, _, event_frame = self.video_files[idx]
        #print(video_path, target, start_frame, end_frame)
        frames, T_diff = self.__load_video(video_path, event_frame = event_frame)
        #print(T_diff)
        frames = torch.stack(frames, dim = 1).permute(1, 0, 2, 3)
        return frames, target, T_diff


if __name__ == '__main__':
    train_dataset = PreAccidentTrainingDataset(
        root_dir = 'dataset/train/train_video',
        csv_file = './dataset/extracted_train.csv',
        num_frames = 16,
        frame_window = 16,
        interested_interval = 100,
        resize_shape = (128, 171),
        crop_size = (112, 112)
    )
    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = 16,
        shuffle = True,
        num_workers = 8,
        pin_memory = True
    )
    for data in data_loader:
        X, y, T = data
        #print(X.shape, y.shape, T.shape)
        #print(T)
          