import torch
import torch.nn as nn
from typing import Tuple, Union
from model_utils.ObjectDetector import ObjectDetector as object_detector
import torch.nn.functional as F 

__all__ = ["LSTM_cell", "DSA_RNN"]
class LSTM_cell(nn.Module):
    def __init__(self, input_size:int, hidden_size:int):
        super(LSTM_cell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Initialize weights for input transformations
        self.W_i = nn.Parameter(torch.randn(hidden_size, input_size ))
        self.W_f = nn.Parameter(torch.randn(hidden_size, input_size ))
        self.W_c = nn.Parameter(torch.randn(hidden_size, input_size ))
        self.W_o = nn.Parameter(torch.randn(hidden_size, input_size ))
        
        # Initialize weights for hidden state transformations
        self.U_i = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.U_f = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.U_c = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.U_o = nn.Parameter(torch.randn(hidden_size, hidden_size))
        
        # Initialize weights for memory cell transformations
        self.V_i = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.V_f = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.V_c = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.V_o = nn.Parameter(torch.randn(hidden_size, hidden_size))
        # Initialize biases
        self.b_i = nn.Parameter(torch.zeros(hidden_size))
        self.b_f = nn.Parameter(torch.zeros(hidden_size))
        self.b_c = nn.Parameter(torch.zeros(hidden_size))
        self.b_o = nn.Parameter(torch.zeros(hidden_size))
        
        
        self.w = nn.Parameter(torch.randn(1, hidden_size//2))
        self.W_e = nn.Parameter(torch.randn(hidden_size//2, hidden_size))
        self.U_e = nn.Parameter(torch.randn(hidden_size//2, hidden_size//2))
        self.b_e = nn.Parameter(torch.zeros(hidden_size//2)) 
         
        self.init_weights() 
    def init_weights(self,):
        for param in self.parameters():
            if len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
    def forward(self, feat: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], state:Tuple[torch.Tensor, torch.Tensor])->Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of LSTM Cell
        
        Args:
            feat: Input tensor of shape (batch_size, object_size, input_size)
            state: Tuple of (h, c) where both shapes are (batch_size, hidden_size)
            
        Returns:
            new_h, new_c: Updated hidden and cell states (batch_size, hidden_size)
        """
        h, c = state
        #h: (batch_size, 1, hidden_size)
        #c: (batch_size, 1, hidden_size)
        
        x_j, x_mask, X_F = feat
        #x_j(object feature): (batch_size, max_num_obj, hidden_size)
        #x_mask(object mask): (batch_size, max_num_obj)
        #X_F(Fullframe feature): (batch_size, hidden_size)
        
        #Attention weight mechanism
        e = torch.matmul(torch.tanh(torch.matmul(h, self.W_e.t()) + torch.matmul(x_j, self.U_e.t()) + self.b_e), self.w.t()).squeeze(2)
        #e(un-normalized attention weight): (batch_size, max_num_obj)
        
        e.masked_fill_(~x_mask, -1e9)
        
        a = torch.softmax(e, dim = 1)
        #a(normalized attention weight): (batch_size, max_num_obj)
        
        phi_j = torch.einsum('ij, ijk -> ik', a, x_j).unsqueeze(1)
        #phi_j: (batch_size, 1, hidden_size)
        
        X_F = X_F.unsqueeze(1)
        #fullframe feature: (batch_size, 1, hidden_size)
        x = torch.cat((phi_j, X_F), dim = 2)
        #x: (batch_size, 1, 2 * hidden_size)

        # Input gate
        i = torch.sigmoid(
            torch.matmul(x, self.W_i.t()) +
            torch.matmul(h, self.U_i.t()) +
            torch.matmul(c, self.V_i.t()) + 
            self.b_i
        )
        #i gate: (batch_size, 1, hidden_feature)
       
        # Forget gate
        f = torch.sigmoid(
            torch.matmul(x, self.W_f.t()) +
            torch.matmul(h, self.U_f.t()) + 
            torch.matmul(c, self.V_f.t()) + 
            self.b_f
        )
        #f gate: (batch_size, 1, hidden_feature)
        new_c = f * c + i * torch.tanh(torch.matmul(x, self.W_c.t()) + torch.matmul(h, self.U_c.t()) + self.b_c)
        #new cell: (batch_size, hidden_feature)
         

        # Output gate
        o = torch.sigmoid(
            torch.matmul(x, self.W_o.t()) +
            torch.matmul(h, self.U_o.t()) +
            torch.matmul(c, self.V_o.t()) + 
            self.b_o
        )
        #o gate: (batch_size, 1, hidden_size)
       
        # New hidden state
        new_h = o * torch.tanh(new_c)
        #new h: (batch_size, 1, hidden_size)
        
        return new_h, new_c

class DSA_RNN(nn.Module):
    def __init__(self, input_size:int = 4096, hidden_size:int = 128, num_layers: int = 1, batch_first = True):
        super(DSA_RNN, self).__init__()
        self.input_size = input_size
        assert self.input_size == 4096 # output dimension gauged by faster r-cnn
        self.hidden_size = hidden_size
        self.batch_first = True
        self.lstm = LSTM_cell(input_size = self.hidden_size, hidden_size = self.hidden_size)
        
        self.object_detector = object_detector(score_threshold = 0.8, max_num_objects = 19)
        
        for param in self.object_detector.parameters():
            param.requires_grad = False
        
        self.object_detector.eval()
        self.fc1 = nn.Linear(hidden_size//2, hidden_size//2)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.bottleneck_obj = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size//2), 
            nn.ReLU(),
            nn.Dropout(0.)
        )
        self.bottleneck_fullframe = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size//2), 
            nn.ReLU(),
            nn.Dropout(0.)
        )
        self.init_weights()
    def combine_features(self, objects_features, objects_mask, fullframe_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        return (self.bottleneck_obj(objects_features), objects_mask, self.bottleneck_fullframe(fullframe_features))
          
    def init_weights(self,):
        for param in self.parameters():
            if len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
    def forward(self, x: torch.Tensor, initial_state: Union[Tuple[torch.Tensor, torch.Tensor], None] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of stacked LSTM
        
        Args:
            x(numerical representation of video): Input tensor of shape (batch_size, n_frames, n_channel, width, height)
            initial_states: Optional tuple of (h_0, c_0) for all layers
                          Each of shape (num_layers, batch_size, hidden_size)
        
        Returns:
            output(torch.Tuple, (batch_size, n_frames)): Sequence of hidden states for each time step.
            h_n(torch.Tuple, (batch_size, hidden_size)): Final hidden states for each layer.
            c_n(torch.Tuple, (batch_size, hidden_size)): Final cell states for each layer.
        """
        batch_size, n_frames = x.shape[0], x.shape[1]
        
        if initial_state is None:
            h_states = torch.zeros((batch_size, 1, self.hidden_size), device = x.device, requires_grad = False)
            c_states = torch.zeros((batch_size, 1, self.hidden_size), device = x.device, requires_grad = False)
        else:
            h_states, c_states = initial_state
        
        out_sequence = []
        if self.batch_first:
            x = x.permute(1, 0, 2, 3, 4)
         
        for t in range(n_frames):
            current_frame = x[t, ...]
            with torch.no_grad():
                _, objects_features, objects_mask, fullframe_features = self.object_detector(current_frame)
            x_t = self.combine_features(objects_features, objects_mask, fullframe_features)
            
            h_states, c_states = self.lstm(
                x_t, (h_states, c_states) 
            )
            score = self.fc2(h_states)
            out_sequence.append(score.squeeze(1))
        output = torch.stack(out_sequence, dim=1) # (batch_size, n_frames)
        return output, (h_states, c_states)