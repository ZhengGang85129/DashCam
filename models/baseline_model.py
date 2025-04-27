import torch
import torch.nn as nn
from torchvision.models.video import r3d_18
from torchvision.models.video import R3D_18_Weights
import os
from typing import List, Optional
os.environ['TORCH_HOME'] = os.getcwd() #will download model weights to your current work directory

class baseline_model(nn.Module):
    def __init__(self, classifier: Optional[List] = None):
        super(baseline_model, self).__init__()
        self.backbone = r3d_18(weights = R3D_18_Weights.DEFAULT)
        if classifier is None:
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)  # Modify output layer for binary classification
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
            self.backbone.fc = nn.Sequential(*_classifier_in_list)
                 
        self.initial_weights()
        self.freeze()
        
         
    def unfreeze_layers(self, layer_names: List[str]) -> None:
        '''
        Unfreezing the parameters in the corresponding parts.
        '''
        
        if "*" in layer_names:
            for param in self.backbone.parameters():
                param.reguires_grad = True
            return True
        else:
            for layer_name in layer_names:
                if not hasattr(self.backbone, layer_name):
                    raise ValueError(f"Layer '{layer_name}' not found in model. Available layers: {list(layer for layer in dir(self.backbone) if not layer.startswith('_'))}")
                layer = getattr(self.backbone, layer_name)

                if not isinstance(layer, torch.nn.Module):
                    raise ValueError(f"The attribute '{layer_name}' exists but is not a nn.Module")
                for param in layer.parameters():
                    param.requires_grad = True

        return
    def freeze(self) -> None:
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        return 
     
    def initial_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear) and name == 'fc':
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        '''
        Args:
        x: [torch.Tensor, (batch_size, n_frames, n_channels, H, W)]
        Return:
        y: [torch.Tensor, (batch_size, num_classes(2))]
        '''
        x = x.permute(0, 2, 1, 3, 4) 
        
        return self.backbone(x)