import pandas as pd
import torch
import torch.nn as nn 
from models.model import baseline_model
from utils.tool import get_device
from utils.Dataset import VideoTo3DImageDataset_Inference
from typing import Optional
import torch.nn.functional as F
import os
from tqdm import tqdm
'''
kaggle competitions submit -c nexar-collision-prediction -f submission.csv -m "Message"
'''
def load_model(state_path: Optional[str] = None) -> nn.Module:
    model = baseline_model()
    if state_path is None:
        raise FileNotFoundError(f"The state_dict is None.")
    elif not os.path.isfile(state_path):
        raise FileNotFoundError(f"The state_dict {state_path} doesn't exist.")
    saved_state = torch.load(state_path, map_location = device if device is not None else 'cpu')

    if isinstance(saved_state, dict):
        if 'model_state_dict' in saved_state:
            model.load_state_dict(saved_state['model_state_dict'])
        else:
            model.load_state_dict(saved_state)
    else:
        model = saved_state
    
    return model

def get_dataloader()-> torch.utils.data.DataLoader:
    '''
    Return:
        test dataloader (torch.utils.)
    '''
    test_dataset = VideoTo3DImageDataset_Inference(
        root_dir="./dataset/test",
        csv_file = './dataset/test.csv',
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size = 16, num_workers = 4, pin_memory = True, shuffle = False
    )
    return test_loader


def main():
    global device, test_dataloader
    device = get_device()
    dataloader = get_dataloader()
    model = load_model(state_path = './first_try/best_model_ckpt_bs16_lr0.001.pt')
    model = model.to(device)
    results = [] 
    
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        X,Ids = batch
        X = X.to(device)
        model.eval()
        with torch.no_grad():
            output = model(X)
        probs = F.softmax(output, dim = 1)[...,1]
        probs = probs.tolist()
        for i, (Id, prob) in enumerate(zip(Ids, probs)):
            results.append({
                "id": Id.item(),
                "score": prob
            })
    submission_df = pd.DataFrame(results)
    submission_df.to_csv('submission.csv', index=False) 

if __name__  == '__main__':
    
    main()