import torch.nn as nn
import torch
from typing import Tuple
from FeatureExtractor import FeatureExtractor as feature_extractor
from ObjectDetector import ObjectDetector as object_detector

class LSTM_cell(nn.Module):
    def __init__(self, input_size:int, hidden_size:int):
        super(LSTM_cell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Initialize weights for input transformations
        self.W_i = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_f = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_c = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_o = nn.Parameter(torch.randn(hidden_size, input_size))
        
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
        
        self.init_weights() 
    def init_weights(self,):
        for param in self.parameters():
            if len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
    def forward(self, x: torch.Tensor, state:Tuple[torch.Tensor, torch.Tensor]):
        """
        Forward pass of LSTM Cell
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            state: Tuple of (h, c) where both shapes are (batch_size, hidden_size)
            
        Returns:
            new_h, new_c: Updated hidden and cell states
        """
        h, c = state
        
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
        
        return new_h, new_c

class DSA_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(DSA_RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.ModuleList(
            [LSTM_cell(input_size = self.input_size if Idx == 0 else self.hidden_size,
                       hidden_size = self.hidden_size) for Idx in range(num_layers)]
        )
        
        self.object_detector = object_detector()
        self.feature_extractor = feature_extractor() 
        
    def forward(self, x: torch.Tensor, initial_state: Tuple[torch.Tensor, torch.Tensor] = None):
        """
        Forward pass of stacked LSTM
        
        Args:
            x: Input tensor of shape (batch_size, n_frames, n_objects, input_size)
            initial_states: Optional tuple of (h_0, c_0) for all layers
                          Each of shape (num_layers, batch_size, hidden_size)
        
        Returns:
            output: Sequence of hidden states for each time step
            (h_n, c_n): Final hidden and cell states for each layer
        """
        batch_size, sequence_length = x.shape[0], x.shape[1]
        
        if initial_state is None:
            h_0 = torch.zeros((batch_size, sequence_length, self.hidden_size), device = x.device)
            c_0 = torch.zeros((batch_size, sequence_length, self.hidden_size), device = x.device)
        else:
            h_0, c_0 = initial_state
        
        h_states = [h_0[i] for i in range(self.num_layers)]
        c_states = [c_0[i] for i in range(self.num_layers)]
        
        out_sequence = []
        
        for t in range(sequence_length):
            x_t = x[:, t, :]
            
            for layer in range(self.num_layers):
                if layer == 0:
                    x_t = x_t
                else:
                    x_t = h_states[layer-1]
                h_states[layer], c_states[layer] = self.lstm[layer](
                    x_t, (h_states[layer], c_states[layer]) 
                )
            
            out_sequence.append(h_states[-1])
        
        output = torch.stack(out_sequence, dim=1)
        
        # Stack final hidden and cell states
        h_n = torch.stack(h_states, dim=0)
        c_n = torch.stack(c_states, dim=0)
        
        return output, (h_n, c_n)