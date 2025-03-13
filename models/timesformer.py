import torch
import torch.nn as nn
from timesformer_pytorch import TimeSformer

class timesformer(nn.Module):
    def __init__(self):
        super(timesformer, self).__init__()
        self.model = TimeSformer(
            dim = 512,
            image_size = 224,
            patch_size = 16,
            num_frames = 16,
            num_classes = 2,
            depth = 12,
            heads = 8,
            dim_head =  64,
            attn_dropout = 0.1,
            ff_dropout = 0.1
        )
        for param in self.model.parameters():
            param.requires_grad = False

        # Then unfreeze only the final layer parameters
        self.model.to_out[1].weight.requires_grad = True
        self.model.to_out[1].bias.requires_grad = True 
        self.model.to_out[0].weight.requires_grad = True
        self.model.to_out[0].bias.requires_grad = True 
    def forward(self, x):
        
        out = self.model(x)
        
        return out


if __name__ == '__main__':
    
    model = timesformer()
    model.eval()
    video = torch.randn(2, 16, 3, 224, 224) # (batch x frames x channels x height x width)
    #mask = torch.ones(2, 8).bool() # (batch x frame) - use a mask if there are variable length videos in the same batch

    #for name, param in model.named_parameters():
    #    print(f"{name}: {param.shape}")
    with torch.no_grad():
        pred = model(video) # (2, 10)
    print(pred)