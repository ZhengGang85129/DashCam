from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
import torch.nn as nn 
import os
from typing import List, Optional


os.environ['TORCH_HOME'] = os.getcwd() 


class mvit_v2(nn.Module):
    def __init__(self, classifier: Optional[List] = None):
        super(mvit_v2, self).__init__()
        
        self.backbone = mvit_v2_s(weights = MViT_V2_S_Weights.DEFAULT)
        
        self.backbone.head = nn.Identity()

        hidden_size = 768

        if classifier is None:
            self.classifier = nn.Linear(hidden_size, 2)
        else:
            _classifier_in_list = []
            prev_infeat = hidden_size 
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
        self.initial_weights()
        self.freeze()
    def freeze(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        return 
    def unfreeze_layers(self, layer_names: List[str]):
        if '*' in layer_names:
            for param in self.backbone.parameters():
                param.requires_grad = True
            for param in self.classifier.parameters():
                param.requires_grad = True
            
            return True
        else:
            for layer_name in layer_names: 
                layer = self.get_layer_by_name(layer_name)
                for param in layer.parameters():
                    param.requires_grad = True
    def get_layer_by_name(self, layer_name: str):
        
        if layer_name == 'classifier':
            return self.classifier
        elif layer_name.startswith('blocks'):
            try:
                block_idx = int(layer_name.split('.')[1])
                layer = self.backbone.blocks[block_idx]
            except(IndexError, ValueError):
                raise ValueError(f'Invalid block index in {layer_name}')
        elif hasattr(self.backbone, layer_name):
            layer = getattr(self.backbone, layer_name)
        else:
            raise ValueError(f"Layer '{layer_name}' not found")
        return layer
    def initial_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear) and name.startswith('classifier'):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4) 
        features = self.backbone(x)
        return self.classifier(features)
        

