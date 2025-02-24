import torch
from torchvision.models import vgg16, VGG16_Weights
import os
import torch.nn as nn
from torchsummary import summary 
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
        
    def forward(self, x):
        print(f'stage 1: {x.shape}')
        x = self.transform(x)
        print(f'stage 2: {x.shape}')
        x  = self.features(x)
        print(f'stage 3: {x.shape}')
        x = torch.flatten(x, 1)
        print(f'stage 4: {x.shape}')
        x = self.classifer(x)
        print(f'stage 5: {x.shape}')
        return x


if __name__ == '__main__':
    model = FeatureExtractor()
    x = torch.zeros([1, 3, 720, 480])
    model(x) 