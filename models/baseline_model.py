import torch
import torch.nn as nn
from torchvision.models.video import r3d_18
from torchvision.models.video import R3D_18_Weights
import os
from typing import List, Optional
os.environ['TORCH_HOME'] = os.getcwd() #will download model weights to your current work directory

class baseline_model(nn.Module):
    def __init__(self, trainable_parts: List = ["*"], classifier: Optional[List] = None):
        super(baseline_model, self).__init__()
        self.model = r3d_18(weights = R3D_18_Weights.DEFAULT)
        if classifier is None:
            self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # Modify output layer for binary classification
        else:
            _classifier_in_list = []
            prev_infeat = self.model.fc.in_features
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
            self.model.fc = nn.Sequential(*_classifier_in_list)
                 
        self.initial_weights()
        self.freeze_parameters() 
        if '*' in trainable_parts:
            self.unfreeze_all_parameters()
        else: 
            for trainable_part in trainable_parts:
                if "classifier" == trainable_part.lower() or "fc" == trainable_part.lower():
                    self.unfreeze_classifer()
                else:
                    self.unfreeze_part( layer_name = trainable_part)
    
    def freeze_parameters(self):
        for param in self.model.parameters():
            param.requires_grad = False
    
    
    def unfreeze_all_parameters(self):
        for param in self.model.parameters():
            param.requires_grad = True
    
    def unfreeze_classifer(self):
        for param in self.model.fc.parameters():
            param.requires_grad = True
    
    def unfreeze_part(self, layer_name: str):
        for name, module in self.model.named_children():
            if name in layer_name:
                for param in module.parameters():
                    param.requires_grad = True
                print(f"Unfroze part: {name}")  
     
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
        
        return self.model(x)