import torch
import torch.nn as nn
from torchvision.models.video import swin3d_s, Swin3D_S_Weights
import os 

os.environ['TORCH_HOME'] = os.getcwd()

class swintransformer(nn.Module):
    def __init__(self):
        super(swintransformer, self).__init__()
        self.model = swin3d_s(weights = Swin3D_S_Weights)
        num_classes = 2
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)
        self.__init__weight()
        
    def __init__weight(self,):
        
        for name, parameters in self.model.named_parameters():
            if 'head.weight' in name:
                 nn.init.xavier_uniform_(parameters)
            elif 'head.bias' in name:
                nn.init.constant_(parameters, 0.0) 
    
    def forward(self, x):
        """
        Args: 
        x (batch_size, n_frames, n_channels, H, W)
        Return: 
        y (batch_size, n_classes)
        """
        x = x.permute(0, 2, 1, 3, 4)
        
        return self.model(x)

if __name__ == "__main__":
    model = swintransformer()
    video = torch.zeros((10, 16, 3, 224, 224))
    #for name, parameter in model.named_parameters():
    #    print(name, parameter) 
    pred = model(video)
    print(pred.shape)