import torch.nn as nn 
import torch
import torch.nn.functional as F
from typing import Union
from tool import set_seed

class AnticipationLoss(nn.Module):
    def __init__(self, frame_of_accident:float = 90, n_frames: float = 100, decay_coefficient: float = 30, device: Union[torch.device, None] = None):
        super(AnticipationLoss, self).__init__()
        self.frame_of_accident = frame_of_accident
        self.n_frames = n_frames
        self.frames = torch.arange(0, n_frames)
        self.f = decay_coefficient 
        pos_penalty = torch.exp(-(F.relu(self.frame_of_accident - self.frames)/self.f))
        neg_penalty = torch.ones([n_frames])
        self.penalty = torch.stack([neg_penalty, pos_penalty], dim = 1) # shape: [n_frames, 2]
        self.ce_loss = nn.CrossEntropyLoss(reduction = 'none')
         
    def forward(self, logits: torch.Tensor, targets: torch.Tensor)->torch.Tensor:
        """
        Args: 
            logit: (batch_size, n_frames, 2): Scores that the corresponding frame has accident or not.
            targets: (batch_size)Indicate whether the corresponding video contains the accident.
        Return:
            Loss [torch.Tensor]: Averaged loss.  
        """
        
        batch_size, n_frames, n_classes = logits.shape
        targets = F.one_hot(targets, num_classes = n_classes)
        targets = targets.unsqueeze(1).expand(-1, n_frames, -1).reshape(-1, n_classes)
        #print(targets.shape)
        logits = logits.reshape(-1, n_classes)
        #print(self.penalty.shape) 
        log_softmax = F.log_softmax(logits)
        #print(log_softmax.shape)
        penalty = self.penalty.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, n_classes)
        #print(penalty.shape) 
        weighted_log_softmax = log_softmax * penalty
        #print(weighted_log_softmax.shape)
        loss = torch.sum(-weighted_log_softmax * targets)/batch_size/n_frames
        #print(loss)
if __name__ == "__main__":
    
    targets = torch.randint(low = 0, high = 2, size = (10,))
    targets = targets.to(torch.long)
    pred = torch.zeros((10, 100, 2))
    Loss = AnticipationLoss()
    print(Loss(pred, targets))
