import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple

class PseduoDataset(Dataset):
    """
    Pseduo dataset for debugging
    """
    def __init__(
        self,
    ):
        
        self.target = torch.randint(0, 2, (1500, ))
        # Get all MP4 files in the directory
        self.dataset = []
        for Index in range(60):
            self.dataset.append(torch.ones(100, 3, 224, 224))

    def __len__(self) -> int:
        return len(self.dataset)


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        
        Args:
            idx (int): Index of video file
        Returns:
            torch.Tensor: Tensor of shape (n_frames, channels, height, width)
        """
        # Stack frames into a single tensor
        return self.dataset[idx].permute(0, 3, 1, 2), self.target[idx]
        