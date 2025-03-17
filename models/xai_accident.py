import torch.nn as nn
import torchvision.models as models
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import os 
os.environ['TORCH_HOME'] = os.getcwd()
#Feature Extractor
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        
        self.model = models.resnet50(weights = models.ResNet50_Weights.IMAGENET1K_V1) # for transfer learning
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 512)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.model(x)
        
#Recurrent neural network
class GRUNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim:int, n_layers:int, dropout =[0, 0]):
        super(GRUNet, self).__init__()
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers = n_layers, batch_first = True)
        self.dense1 = nn.Linear(hidden_dim, 64)
        self.dense2 = nn.Linear(64, output_dim)
        #self.relu = nn.ReLU()
        self.dropout = dropout
        self.__init__weight()
         
    def __init__weight(self):
        
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        nn.init.xavier_normal_(self.dense1.weight)
        nn.init.zeros_(self.dense1.bias)
        nn.init.xavier_normal_(self.dense2.weight)
        nn.init.zeros_(self.dense2.bias)
    def forward(self, x, h) -> torch.Tensor:
        
        out, h = self.gru(x, h)
        out = F.dropout(out[:, -1], self.dropout[0])
        out = F.relu(self.dense1(out))
        out = F.dropout(out, self.dropout[1])
        out = self.dense2(out)
        
        return out, h


class AccidentXai(nn.Module):
    def __init__(self, num_classes: int = 2,  h_dim: int = 256, n_layers: int = 1):
        
        super(AccidentXai, self).__init__()
        
        self.h_dim = h_dim
        self.n_layers = n_layers 
        self.num_classes = num_classes
        
        self.ft_extractor = FeatureExtractor()
        self.gru_net = GRUNet(h_dim + h_dim, h_dim, self.num_classes, n_layers, dropout = [0.2, 0.0])
    
    def forward(self, x:torch.Tensor, ) -> torch.Tensor:
        
        all_outputs = []
        _, T, _, _, _ = x.shape
        h = Variable(torch.zeros(self.n_layers, x.size(0), self.h_dim))
        h = h.to(x.device)
        
        for t in range(T):
            x_t = self.ft_extractor(x[:, t])
            x_t = torch.unsqueeze(x_t, 1)
            
            output, h = self.gru_net(x_t, h)
            all_outputs.append(output)
        
        return torch.stack(all_outputs, dim = 1)


if __name__ == "__main__":
    
    model = AccidentXai(2, h_dim = 256, n_layers = 2)
    X = torch.tensor(torch.zeros(10, 16, 3, 224, 224))
    pred = model(X)
    print(pred.shape)