import torch
import torch.nn as nn
from torchvision.models.video import r3d_18
from torchvision.models.video import R3D_18_Weights
import os
from typing import List, Optional
os.environ['TORCH_HOME'] = os.getcwd() #will download model weights to your current work directory

class LayerNorm3D(nn.Module):
    def __init__(self, num_features):
        super(LayerNorm3D, self).__init__()
        self.layernorm = nn.LayerNorm(num_features)
    def forward(self, x):
        
        b, c, d, h, w = x.size()
        x = x.view(b, c, -1)
        x = x.permute(0, 2, 1)
        x = self.layernorm(x)
        x = x.permute(0, 2, 1)
        x = x.view(b, c, d, h, w)
        return x


class baseline_model(nn.Module):
    def __init__(self, classifier: Optional[List] = None):
        super(baseline_model, self).__init__()
        self.backbone = r3d_18(weights = R3D_18_Weights.DEFAULT)
        
        backbone_final_in = self.backbone.fc.in_features  
        self.backbone.fc = nn.Identity()
        if classifier is None:
            self.classifier = nn.Linear(backbone_final_in, 2)  # Modify output layer for binary classification
        else:
            _classifier_in_list = []
            prev_infeat = backbone_final_in
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
                 
        #self.replace_batchnorm_with_layernorms(self.backbone)
        self.initial_weights()
        self.freeze()
    
    def replace_batchnorm_with_layernorms(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm3d):
                num_features = child.num_features
                setattr(module, name, LayerNorm3D(num_features))
            else:
                self.replace_batchnorm_with_layernorms(child) 
      
    def get_layer_by_name(self, layer_name):
        
        if hasattr(self.backbone, layer_name):
            return getattr(self.backbone, layer_name)
        elif layer_name == 'classifier':
            return self.classifier
        else:
             raise ValueError(f"Layer '{layer_name} not found in model")
    
       
    def unfreeze_layers(self, layer_names: List[str]) -> None:
        '''
        Unfreezing the parameters in the corresponding parts.
        '''
        
        if "*" in layer_names:
            for param in self.backbone.parameters():
                param.requires_grad = True
            return True
        else:
            for layer_name in layer_names:
                if layer_name == 'classifier':
                    layer = self.classifier
                elif hasattr(self.backbone, layer_name):
                    layer = getattr(self.backbone, layer_name)
                else:
                    available = ['classifier'] + [
                    name for name in dir(self.backbone)
                    if isinstance(getattr(self.backbone, name), nn.Module) and not name.startswith("_")
                    ]
                    raise ValueError(f"Layer '{layer_name}' not found. Available: {available}")

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
            if isinstance(m, nn.Linear) and name.startswith('classifier'):
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
        feat = self.backbone(x)
        out = self.classifier(feat) 
        return out