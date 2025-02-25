import sys, os 
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from model  import LSTM_cell, DSA_RNN # type: ignore
import torch
from dataset import VideoDataset


feat = (torch.randn((1, 20, 4096)), torch.randn((1, 1, 4096)))
state = (torch.randn((1, 1, 256)), torch.randn((1, 1, 256)))


if __name__ == "__main__":
    
    model = DSA_RNN(input_size = 4096, hidden_size = 512, num_layers = 1)
    
    dataset = VideoDataset()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = 4,
        shuffle = True,
        num_workers = 4
    )
    
    for data in dataloader:
        n_frames = data.shape[1]
        print('x -> ', data.shape)
        output, _ = model(x = data)
        print(output)
        break
        #model(data) 