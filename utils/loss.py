import torch.nn as nn 
import torch
import torch.nn.functional as F
from typing import Union
import numpy as np 
class AnticipationLoss(nn.Module):
    def __init__(self, frame_of_accident:float = 90, n_frames: float = 100, decay_coefficient: float = 30, device: Union[torch.device, None] = None, gamma: float = 1):
        super(AnticipationLoss, self).__init__()
        self.frame_of_accident = frame_of_accident
        self.n_frames = n_frames
        self.frames = torch.arange(0, n_frames)
        self.f = decay_coefficient 
        pos_penalty = torch.exp(-(F.relu(self.frame_of_accident - self.frames + 15)/self.f))
        neg_penalty = torch.ones([n_frames])
        self.penalty = torch.stack([neg_penalty, pos_penalty], dim = 1) # shape: [n_frames, 2]
        self.ce_loss = nn.CrossEntropyLoss(reduction = 'none')
        self.gamma = gamma 
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
        log_softmax = F.log_softmax(logits, dim = -1)
        #print(log_softmax.shape)
        penalty = self.penalty.unsqueeze(0).expand(batch_size, -1, -1).reshape(-1, n_classes)
        #print(penalty.shape) 
        weighted_log_softmax = log_softmax * penalty.to(logits.device)
        #print(weighted_log_softmax.shape)
        loss = torch.sum(-weighted_log_softmax * targets)/batch_size/n_frames
        return loss
        #print(loss)

class TemporalBinaryCrossEntropy(nn.Module):
    
    def __init__(self, decay_coefficient:int = 20, alpha: float = 1):
        super(TemporalBinaryCrossEntropy, self).__init__()
         
        self.decay_coefficient = decay_coefficient
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', )
        self.alpha = alpha
        self.Lambda = alpha/self.expected_value(decay_coefficient)  
    def expected_value(self,T)->float:
        x = np.arange(0, 46)
        expectation = np.mean(np.exp(-x / T))
        return expectation
    
    def forward(self, logit: torch.Tensor, target: torch.Tensor, T_diff: torch.Tensor) -> torch.Tensor:
        '''
        logit: (N, 2)
        target: (N)
        T_diff: (N)
        ''' 
        pos_weight = self.Lambda*torch.exp(-F.relu(T_diff+15)/self.decay_coefficient).to(target.device)
        neg_weight = torch.ones_like(target).to(target.device)
        one_hot = F.one_hot(target, num_classes = 2).to(logit.device)
        weight = torch.stack([neg_weight, pos_weight], dim = 1).to(logit.device)
        probs = F.softmax(logit, dim = 1)
        sample_weights = torch.sum(one_hot * weight, dim=1)
        
        #print(target, output)
        per_sample_loss = self.ce_loss(logit, target)
        #print(per_sample_loss)
        #print(sample_weights)
        #print(sample_weights * per_sample_loss)
        
        return  torch.mean(sample_weights * per_sample_loss )


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        :param alpha: 類別權重，可以是 float 或 tensor
        :param gamma: 聚焦參數
        :param reduction: 'mean' | 'sum' | 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        :param inputs: [B, C] logits
        :param targets: [B] 真實標籤 (long)
        """
        log_probs = F.log_softmax(inputs, dim=1)  # 轉成 log-probs
        probs = torch.exp(log_probs)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()

        focal_weight = (1 - probs) ** self.gamma
        focal_weight = focal_weight * targets_one_hot

        loss = -self.alpha * focal_weight * log_probs
        loss = loss.sum(dim=1)  # sum over classes

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

if __name__ == "__main__":
    Loss_fn = TemporalBinaryCrossEntropy(decay_coefficient=20)
    output = torch.tensor([-5e-2, 1]).expand([16, -1])
    #torch.ones(size = (16, 2)).to(torch.float32)
    target = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,1, 1,1 ,1 ,1]).to(torch.long)
    T_diff = torch.arange(start = 6, end = -10, step = -1)
    #target = torch.randint(low = 0, high = 2, size=(16, )).to(torch.long)
    #end_frame = torch.randint(low = 0, high = 96, size = (16, 1))
    
    print(output.shape, target)
    
    print(Loss_fn(output, target, T_diff))
    
