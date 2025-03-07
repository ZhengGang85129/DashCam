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
        #self.classifer = nn.Sequential(
        #    *list(model.classifier.children())[:-1]
        #) 
        self.transform = VGG16_Weights.DEFAULT.transforms()
        for param in self.features.parameters():
            param.requires_grad = False
        self.classifer = nn.Sequential(*list(model.classifier.children())[:-1]) 
        self.eval()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
        x(torch.Tensor, batch_size, channel, width, height)
        Args:
        x(torch.Tensor, batch_size, 4096, 224, 224)
        '''
        x = self.transform(x)
        x = self.features(x)
        x = torch.flatten(x)
        x = self.classifer(x)
        x = x.squeeze(0)
        return x


if __name__ == "__main__":

    x = torch.ones((5, 3, 224, 224))
    model = FeatureExtractor()
    out = model(x)
    print(out.shape)