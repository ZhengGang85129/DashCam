import torch
import torch.nn as nn
from torchvision.models.video import r3d_18
from torchvision.models.video import R3D_18_Weights
import os

os.environ['TORCH_HOME'] = os.getcwd() #will download model weights to your current work directory

class baseline_model(nn.Module):
    def __init__(self):
        super(baseline_model, self).__init__()
        self.model = r3d_18(weights = R3D_18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # Modify output layer for binary classification
    def forward(self, x:torch.Tensor)->torch.Tensor:
        '''
        Args:
            x (torch.Tensor): batch_size, n_frames, n_channels, height, width
        Return:
            y (torch.Tensor): batch_size, n_classes
        '''
        x = x.permute(0, 2, 1, 3, 4)
        return self.model(x)

if __name__ == "__main__":
    
    model = baseline_model()
    print(model)