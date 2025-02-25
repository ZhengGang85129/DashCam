import torch.nn as nn 
import torch
class AnticipationLoss(nn.Module):
    def __init__(self, decay_nframe = 50, pivot_frame_index = 90):
        super(AnticipationLoss, self).__init__()
        self.decay_nframe = decay_nframe
        self.pivot_frame_index = pivot_frame_index 
    def forward(self, prob, targets)->torch.Tensor:
        y = targets.unsqueeze(1)
        t_step = torch.arange(start = 0, end = prob.shape[1]).unsqueeze(0)#.expand((prob.shape[0], 1))
        loss = - y * torch.exp(- torch.relu(self.pivot_frame_index - t_step)/self.decay_nframe) * torch.log(prob) - (1 - y) * torch.log(prob) 
        loss = loss.mean() 
        return loss 

if __name__ == "__main__":
    
    output_prob = torch.rand((4, 100))
    target = torch.ones((4))
     
    Loss = AnticipationLoss()
    Loss(output_prob, target) 