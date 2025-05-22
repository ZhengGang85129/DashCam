import torch
import torch.nn as nn
from torchvision.models.video import swin3d_s, Swin3D_S_Weights
import os 
from typing import List, Optional

os.environ['TORCH_HOME'] = os.getcwd()

class swintransformer(nn.Module):
    def __init__(self, classifier: Optional[List] = None):
        super(swintransformer, self).__init__()
        self.backbone = swin3d_s(weights = Swin3D_S_Weights)
        
        backbone_final_in = self.backbone.head.in_features
        self.backbone.head = nn.Identity()
        
        if classifier is None:
            self.classifier = nn.Linear(backbone_final_in, 2)
        else:
            _classifier_in_list = []
            prev_infeat = self.backbone.fc.in_features
            for elem in classifier:
                if isinstance(elem, str):
                    if elem.lower() == 'relu':
                        _classifier_in_list.append(nn.ReLU())
                    elif 'dropout' in elem.lower():
                        dropout = elem.lower().split(':')
                        _classifier_in_list.append(nn.Dropout(p = float(dropout[1])))
                elif isinstance(elem, int):
                    _classifier_in_list.append(nn.Linear(prev_infeat, elem))
                    prev_infeat = elem
                else:
                    raise ValueError(f"Can't assign such layer: {elem}") 
            _classifier_in_list.append(nn.Linear(prev_infeat, 2))
            self.classifier = nn.Sequential(*_classifier_in_list) 
         
        self.__init__weight()
    
    def unfreeze_layers(self, layer_names: List[str]) -> None:
        
        
        
        if '*' in layer_names:
            for param in self.backbone.parameters():
                param.requires_grad = True
            return True
        else:
            for layer_name in layer_names:
                layer = self.get_layer_by_name(layer_name) 
                for param in layer.parameters():
                    param.requires_grad = True 
                    print(param)
                print(f'[Unfreeze]: {layer_name}')
                 
    def get_layer_by_name(self, layer_name):
        
        swin_map = {
            'layer1': 0,
            'layer2': 1,
            'layer3': 2,
            'layer4': 3,
        }
        if layer_name in swin_map:
            return self.backbone.features[swin_map[layer_name]]
        elif layer_name == 'classifier':
            return self.classifier
        raise ValueError(f"Layer '{layer_name} not found in model") 
     
    def __init__weight(self,):
        
        for name, m in self.named_parameters():
            if isinstance(m, nn.Linear) and name.startswith('classifier'):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def freeze(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = False
        return 

    
    def forward(self, x):
        """
        Args: 
        x (batch_size, n_frames, n_channels, H, W)
        Return: 
        y (batch_size, n_classes)
        """
        x = x.permute(0, 2, 1, 3, 4)
        feat = self.backbone(x)
        return self.classifier(feat)

if __name__ == "__main__":
    model = swintransformer()
    video = torch.zeros((10, 16, 3, 224, 224))
    for name, parameter in model.named_parameters():
        print(name,) 