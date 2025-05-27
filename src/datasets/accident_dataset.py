import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict, Optional
import pandas as pd
import os 
import random
from PIL import Image
from pathlib import Path
from src.datasets.transforms import CustomTransforms
import cv2



class AccidentDataset(Dataset):
    '''
    The sampling approach is the sliding window.
    '''
    def __init__(
        self,
        root_dir: str = 'dataset/img_database/train',
        csv_file: str = 'dataset/frame_train.csv',
        mode: str = 'evaluation',
        model_type: str = 'baseline',
        stride: int = 2,
        frame_per_window: int = 16
    ):
        self.root_dir = Path(root_dir)
        self.metadata = pd.read_csv(csv_file)
        #self.metadata = self.metadata.drop(index = 0)
         
        self.frame_per_window = frame_per_window
        self.stride = stride 
        self.transform = CustomTransforms(model_type) 
        self.mode = mode 
        
        assert mode in ['training', 'evaluation', 'validation'], 'mode: {self.mode} is not supported.'       
        if self.mode == 'training' or self.mode == 'validation':
            self.get_vids = self.metadata.video_id.unique().tolist()
        else:
            self.get_vids = self.metadata.id.unique().tolist() 
        
        

    def __len__(self) -> int:
        return len(self.get_vids)
    
    def __get_training_clips(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        '''
        Args:
            idx(index): global index in metadata frame.
        Return:
            frames(torch.Tensor): [n_frames, n_channels, height, width]
            label (int)
            T_diff (int)
            concerend(bool)
        ''' 
        row = self.metadata[self.metadata.video_id == self.get_vids[idx]]

        if row.label.sum()>0:
            row = row[row.positive == 1]
            row = row[(row.last_frame - row.frame >= 15) & (row.last_frame - row.frame <= 45)]
        while True:
            random_row = row.sample(n = 1)
            if random_row.frame.item() >= (self.frame_per_window - 1) * self.stride:
                break 
        
        folder = self.root_dir / Path(f'video_{random_row.video_id.item():05d}')
        
        start = random_row.frame.item() - self.stride * (self.frame_per_window - 1)
        
        end = random_row.frame.item() + self.stride
        
        img_paths = [folder/Path(f'{img_index:05d}.jpg') for img_index in range(start, end, self.stride)]
        frames = []
        
        for img_path in img_paths:
            if os.path.exists(img_path):
                img = cv2.imread(str(img_path))
                if img is not None:
                    frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                else:
                    print(f"WARNING: Failed to read image {img_path}")
            else:
                print(f"Warning: File not found {img_path}")

        frames = self.transform.get_transforms(frames, mode = self.mode)
        if len(frames) == 0:
            raise ValueError()
        
        frames = torch.stack(frames, dim = 0)
        return frames, random_row.label.item(), random_row.T_diff.item(), random_row.weight.item(), random_row.positive.item()   
    
    def __get_inferece_clips(self, idx: int) -> Tuple[torch.Tensor, int]:
        '''
        Args:
            idx(index): global index in metadata frame.
        Return:
            Clips(torch.Tensor): (n_frames, n_channels, height, width)
            video id (int)
        ''' 
        row = self.metadata.iloc[idx]
        folder = self.root_dir / Path(f'video_{row.id.item():05d}')
        file_count = len([f for f in os.listdir(folder) if f.endswith('.jpg')]) 
        
        end = file_count
        start = end - self.frame_per_window * self.stride
         
        img_paths = [folder/Path(f'{img_index:05d}.jpg') for img_index in range(start, end, self.stride)]
        frames = []
        
        for img_path in img_paths:
            if os.path.exists(img_path):
                img = cv2.imread(str(img_path))
                if img is not None:
                    frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                else:
                    print(f"WARNING: Failed to read image {img_path}") 
            else:
                print(f"WARNING: File not found {img_path}") 
        frames = self.transform.get_transforms(frames, mode = self.mode) 
        if len(frames) != self.frame_per_window:
            raise ValueError()
        frames = torch.stack(frames, dim = 0) 
        return frames, row.id.item() 
    
    def __getitem__(self, idx: int):
        """
        """ 
        if self.mode == 'training' or self.mode == 'validation':
            return self.__get_training_clips(idx = idx)  
        elif self.mode == 'evaluation':
            return self.__get_inferece_clips(idx = idx) 


def main():
    # Your logic here
    print("Running accident_dataset module")
    datasets = AccidentDataset(
        root_dir = 'dataset/img_database/validation',
        csv_file = 'dataset/frame_validation.csv',
        model_type = 'mvit_v2',
        stride = 3,
        frame_per_window = 16,
        mode = 'validation')
    dataloader = torch.utils.data.DataLoader(datasets,      
            batch_size= 4,
            num_workers = 6,
            pin_memory = False,)
    
    for batch in dataloader:
        frames, target, *_ = batch
if __name__ == "__main__":
    main() 
        

