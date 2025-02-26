import torch.nn as nn 
import torch
class AnticipationLoss(nn.Module):
    def __init__(self, decay_nframe = 50, pivot_frame_index = 90):
        super(AnticipationLoss, self).__init__()
        self.decay_nframe = decay_nframe
        self.pivot_frame_index = pivot_frame_index 
    def forward(self, prob: torch.Tensor, targets: torch.Tensor)->torch.Tensor:
        """
        Args: 
            prob[torch.Tensor, (batch_size, n_frames)]: Scores that the corresponding frame has accident or not.
            targets[torch.Tensor, (batch_size)]: Indicate whether the corresponding video contains the accident.
        Return:
            Loss [torch.Tensor]: Averaged loss.  
        """
        y = targets.unsqueeze(1)
        t_step = torch.arange(start = 0, end = prob.shape[1]).unsqueeze(0).to(prob.device) 
        loss = - y * torch.exp(- torch.relu(self.pivot_frame_index - t_step)/self.decay_nframe) * torch.log(prob) - (1 - y) * torch.log(prob) 
        loss = loss.mean().to(prob.device) 
        return loss 

if __name__ == "__main__":
    
    output_prob = torch.rand((4, 100))
    target = torch.ones((4))
    Loss = AnticipationLoss()
    print(Loss(output_prob, target))