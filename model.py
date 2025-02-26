import torch.nn as nn
import torch
from typing import Tuple, Union
from ObjectDetector import ObjectDetector as object_detector

__all__ = ["LSTM_cell", "DSA_RNN"]
class LSTM_cell(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, scale_input = True):
        super(LSTM_cell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Initialize weights for input transformations
        scale = 2 if scale_input else 1
        self.W_i = nn.Parameter(torch.randn(hidden_size, input_size * scale))
        self.W_f = nn.Parameter(torch.randn(hidden_size, input_size * scale))
        self.W_c = nn.Parameter(torch.randn(hidden_size, input_size * scale))
        self.W_o = nn.Parameter(torch.randn(hidden_size, input_size * scale))
        
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
        
        
        self.w = nn.Parameter(torch.randn(1, hidden_size))
        self.W_e = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.U_e = nn.Parameter(torch.randn(hidden_size, input_size))
        self.b_e = nn.Parameter(torch.zeros(hidden_size)) 
         
        self.init_weights() 
    def init_weights(self,):
        for param in self.parameters():
            if len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
    def forward(self, feat: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], state:Tuple[torch.Tensor, torch.Tensor]):
        """
        Forward pass of LSTM Cell
        
        Args:
            feat: Input tensor of shape (batch_size, object_size, input_size)
            state: Tuple of (h, c) where both shapes are (batch_size, hidden_size)
            
        Returns:
            new_h, new_c: Updated hidden and cell states
        """
        h, c = state
        
        x_j, x_mask, X_F = feat 
        #Attention weight mechanism
        #print('h: ', h.shape, ', c: ', c.shape, 'W_e: ', self.W_e.shape, ' x_j: ', x_j.shape)
        e = torch.matmul(torch.tanh(torch.matmul(h, self.W_e.t()) + torch.matmul(x_j, self.U_e.t()) + self.b_e), self.w.t()).squeeze(2)
        e.masked_fill_(~x_mask, -1e9)
        a = torch.softmax(e, dim = 1)
        x_j = torch.einsum('ij, ijk -> ik', a, x_j).unsqueeze(1)
        
        X_F = X_F.unsqueeze(1)
        x = torch.concat((x_j, X_F), dim = 2) 
        # Input gate
        i = torch.sigmoid(
            torch.matmul(x, self.W_i.t()) +
            torch.matmul(h, self.U_i.t()) +
            torch.matmul(c, self.V_i.t()) + 
            self.b_i
        )
        # Forget gate
        f = torch.sigmoid(
            torch.matmul(x, self.W_f.t()) +
            torch.matmul(h, self.U_f.t()) + 
            torch.matmul(c, self.V_f.t()) + 
            self.b_f
        )
        
        new_c = f * c + i * torch.tanh(torch.matmul(x, self.W_c.t()) + torch.matmul(h, self.U_c.t()) + self.b_c)
         

        # Output gate
        o = torch.sigmoid(
            torch.matmul(x, self.W_o.t()) +
            torch.matmul(h, self.U_o.t()) +
            torch.matmul(c, self.V_o.t()) + 
            self.b_o
        )
        
        # New hidden state
        new_h = o * torch.tanh(new_c)
        #print('new_h: ', new_h.shape)
        #print('new_c: ', new_c.shape)
        
        return new_h, new_c

class DSA_RNN(nn.Module):
    def __init__(self, input_size:int = 4096, hidden_size:int = 128, num_layers: int = 1):
        super(DSA_RNN, self).__init__()
        self.input_size = input_size
        assert self.input_size == 4096 # output dimension gauged by faster r-cnn
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.ModuleList(
            [LSTM_cell(input_size = self.input_size if Idx == 0 else self.hidden_size,
                       hidden_size = self.hidden_size) for Idx in range(num_layers)]
        )
        
        self.object_detector = object_detector(score_threshold = 0.8, max_num_objects = 20)
        
        for param in self.object_detector.parameters():
            param.requires_grad = False
        
        self.object_detector.eval()
        self.fc = nn.Linear(hidden_size, 1)
         
    def forward(self, x: torch.Tensor, initial_state: Union[Tuple[torch.Tensor, torch.Tensor], None] = None):
        """
        Forward pass of stacked LSTM
        
        Args:
            x(numerical representation of video): Input tensor of shape (batch_size, n_frames, n_channel, width, height)
            initial_states: Optional tuple of (h_0, c_0) for all layers
                          Each of shape (num_layers, batch_size, hidden_size)
        
        Returns:
            output: Sequence of hidden states for each time step
            (h_n, c_n): Final hidden and cell states for each layer
        """
        batch_size, n_frames = x.shape[0], x.shape[1]
        
        if initial_state is None:
            h_0 = torch.zeros((batch_size,1, self.hidden_size), device = x.device)
            c_0 = torch.zeros((batch_size, 1, self.hidden_size), device = x.device)
        else:
            h_0, c_0 = initial_state
        assert self.num_layers == 1 #FIX ME : Can only handle one layer now 
        h_states = [h_0 for _ in range(self.num_layers)]
        c_states = [c_0 for _ in range(self.num_layers)]
        
        out_sequence = []
        
        for t in range(n_frames):
            current_frame = x[:, t, :, :, :]
            with torch.no_grad():
                _, objects_features, objects_mask, fullframe_features = self.object_detector(current_frame)
                
            for layer in range(self.num_layers): #FIX can only handle one layer
                if layer == 0: 
                    x_t = (objects_features, objects_mask, fullframe_features)
                else:
                    x_t = h_states[layer-1]
                
                h_states[layer], c_states[layer] = self.lstm[layer](
                    x_t, (h_states[layer], c_states[layer]) 
                )
            out_sequence.append(torch.sigmoid(self.fc(h_states[-1])).squeeze(-1).squeeze(-1))
        output = torch.stack(out_sequence, dim=1) # (batch_size, n_frames)
        # Stack final hidden and cell states
        h_n = torch.stack(h_states, dim=0) # (batch_size, n_frames, hidden_size)
        c_n = torch.stack(c_states, dim=0) # (batch_size, n_frames, hidden_size)
        
        return output, (h_n, c_n)