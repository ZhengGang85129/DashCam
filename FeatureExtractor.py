import torch
from torchvision.models import vgg16, VGG16_Weights
import os
import torch.nn as nn
from torchvision.transforms import transforms

os.environ['TORCH_HOME'] = os.getcwd() #will download model weights to your current work directory

class FeatureExtractor(nn.Module):
    def __init__(self):
        
        super(FeatureExtractor, self).__init__()
        model = vgg16(weights = VGG16_Weights.DEFAULT) 
        
        self.features = model.features
        self.classifer = nn.Sequential(
            *list(model.classifier.children())[:-1]
        ) 
        self.transform = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.CenterCrop(size = [224, 224]),
            transforms.Normalize(mean = [0.485, 0.465, 0.406],
                                std = [0.229, 0.224, 0.225]
                                )
        ])
        
         
        self.eval()
        
    def forward(self, x: torch.Tensor):
        '''
        Args:
        x(torch.Tensor, batch_size, channel, width, height)
        '''
        x = x.unsqueeze(0)
        x = self.transform(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifer(x)
        x = x.squeeze(0)
        return x

