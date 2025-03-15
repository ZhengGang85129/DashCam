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
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            #nn.Dropout(p = 0.1),
            nn.Linear(64, 2)
        )
        self.__init_weight()
        for name, param in self.model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
    
    def __init_weight(self,)->None:
        for m in self.model.fc.modules():
            if isinstance(m, nn.Linear):
                # Use Kaiming/He initialization for ReLU activations
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
      
    def forward(self, x:torch.Tensor)->torch.Tensor:
        '''
        Args:
            x (torch.Tensor): batch_size, n_frames, n_channels, height, width
        Return:
            y (torch.Tensor): batch_size, n_classes
        '''
        x = x.permute(0, 2, 1, 3, 4 ) #https://github.com/pytorch/vision/tree/5a315453da5089d66de94604ea49334a66552524/torchvision/models/video
        return self.model(x)

if __name__ == "__main__":
    
    model = baseline_model()
    #print(model)
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(name)