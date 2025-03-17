from timesformer_pytorch import TimeSformer
import torch
import torch.nn as nn
import os

os.environ['TORCH_HOME'] = os.getcwd()

class timesformer(nn.Module):
    def __init__(self, ):
        super(timesformer, self).__init__()
        
        # Do not change the following hyperparam
        self.model = TimeSformer(
            dim = 512,
            image_size = 224, 
            patch_size = 16,
            num_classes = 2,
            depth = 12,
            heads = 8,
            dim_head = 64,
            attn_dropout = 0.1,
            ff_dropout = 0.1,
            num_frames = 16 
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
        x: [torch.Tensor, (batch_size, n_frames, n_channels, H, W)]
        Return:
        y: [torch.Tensor, (batch_size, num_classes(2))]
        '''
        
        return self.model(x)
        