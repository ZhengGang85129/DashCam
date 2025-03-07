import torch.nn as nn 
import torch
import torch.nn.functional as F
from typing import Union
from tool import set_seed

class AnticipationLoss(nn.Module):
    def __init__(self, decay_nframe:float = 10, pivot_frame_index:float = 100, f2: float = 150, n_frames: int = 100, device: Union[torch.device, None] = None):
        super(AnticipationLoss, self).__init__()
        pos_weights = torch.exp(-F.relu(pivot_frame_index  - torch.arange(0, n_frames) - 1) / decay_nframe).view(-1, 1)
        neg_weights = torch.ones((n_frames, 1))
        self.n_frames = n_frames
        # (n_frames x 2)
        self.frame_weights = torch.cat([neg_weights, pos_weights], dim=1).unsqueeze(1)
        self.frame_weights = self.frame_weights.to(device)
        self.frame_weights.requires_grad = False
        self.log_softmax = torch.nn.LogSoftmax(dim=2)
        self.nll_loss = torch.nn.NLLLoss()
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor)->torch.Tensor:
        """
        Args: 
            logit: (batch_size, n_frames, 2): Scores that the corresponding frame has accident or not.
            targets: (batch_size)Indicate whether the corresponding video contains the accident.
        Return:
            Loss [torch.Tensor]: Averaged loss.  
        """
        logits = logits.permute(1, 0, 2) # (n_frames, batch_size, 2)
        loss = self.log_softmax(logits) # (n_frames, batch_size, 2)
        loss = torch.mul(loss, self.frame_weights) #@ loss
        loss = loss.view(-1, 2)
        labels = targets.unsqueeze(1).expand(-1, self.n_frames).reshape(-1).long()
        loss = self.nll_loss(loss, labels)
        return loss

if __name__ == "__main__":
    set_seed()    
    output_prob = torch.exp(-torch.ones((16, 100, 2))/0.1)
    target = torch.ones((16))
    Loss = AnticipationLoss()
    print(Loss(output_prob, target))
