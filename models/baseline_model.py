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
        return self.model(x)