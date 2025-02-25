import torch
import os
import torch
from torch.utils.data import Dataset
import cv2
from typing import List, Tuple
import pandas as pd
class VideoDataset(Dataset):
    """
    Custom dataset for loading video files from a directory.
    """
    def __init__(
        self,
        root_dir: str = "./dataset/train/extracted",
        csv_file: str = './dataset/train.csv',
        #transform: Optional[transforms.Compose] = None,
        resize_shape: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize the video dataset.
        Args:
            root_dir (str): Directory containing the video files
            transform (transforms.Compose, optional): Transform to be applied to frames
            resize_shape (tuple): Target size for frame resizing (height, width)
        """
        self.root_dir = root_dir
        self.resize_shape = resize_shape
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
            frame = cv2.resize(frame, self.resize_shape)
            # Apply transforms
            frames.append(torch.from_numpy(frame))
                
            frame_count += 1
            
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
# Example usage:
if __name__ == "__main__":
    # Create dataset
    dataset = VideoDataset(
        root_dir="./dataset/train/extracted",
        resize_shape=(224, 224)  # Resize frames to 224x224
    )
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4
    )
    
    # Example iteration
    for batch in dataloader:
        # batch shape: (batch_size, clip_len, channels, height, width)
        video, target = batch
        print(video.shape, target.shape)
        break