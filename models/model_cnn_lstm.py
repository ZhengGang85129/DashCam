import torch
import torch.nn as nn
from typing import Tuple, Union
import torchvision.models as models

class CNN_LSTM(nn.Module):
    def __init__(self, input_size=300, hidden_size=256, num_layers=3, batch_first=True):
        super(CNN_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        # ResNet-101 Feature Extractor
        self.resnet = models.resnet101(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, input_size)
        # Freeze ResNet parameters
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.eval()

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first
        )

        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2)
        )

    def forward(self, x: torch.Tensor, initial_state: Union[Tuple[torch.Tensor, torch.Tensor], None] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, n_frames = x.size(0), x.size(1)

        # Extract features for each frame using ResNet
        features = []
        for t in range(n_frames):
            frame = x[:, t, :, :, :]  # (batch_size, channels, height, width)
            with torch.no_grad():
                frame_feature = self.resnet(frame)  # (batch_size, input_size)
            features.append(frame_feature.unsqueeze(1))
        features = torch.cat(features, dim=1)  # (batch_size, n_frames, input_size)

        # Pass through LSTM
        if initial_state is None:
            out, (hn, cn) = self.lstm(features)
        else:
            out, (hn, cn) = self.lstm(features, initial_state)

        # Classify each time step
        scores = self.fc(out)  # (batch_size, n_frames, 2)

        return scores, (hn, cn)
