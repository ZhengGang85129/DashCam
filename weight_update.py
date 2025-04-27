import torch
from experiment.clip_dataset import TrainingClipDataset
import pandas as pd
import os 
import tqdm 
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from utils.misc import parse_args 
from utils.tool import get_device, set_seed
from models.model import get_model
from torch.amp import autocast, GradScaler
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
from utils.loss import TemporalBinaryCrossEntropy
set_seed(123)

device = get_device()
model = get_model('baseline')()
loss_fn = TemporalBinaryCrossEntropy(30)
model.load_state_dict(torch.load('./random_sampling/best_model_ckpt_bs10_lr1e-4_aug0.25_col_fog_gau_hor_noi_rai.pt'))

model.to(device)

dataset = TrainingClipDataset(
    root_dir = './dataset/image/train/',
    csv_file = 'dataset/frame_train.csv',
    num_frames = 16,
    frame_window = 16,
    resize_shape = (128, 171),
    crop_size = (112, 112),
    inference = True
)


Loader = DataLoader(
    dataset,
    batch_size = 32,
    num_workers = 8
)

scores = []
truths = []
frame_ids = [] # to store the frame index
loss = 0

with torch.no_grad():
    model.eval()
    for i, data in tqdm.tqdm(enumerate(Loader), total = len(Loader)):
        X, target, T_diff, fid = data
        X = X.to(device)
        target = target.to(device)
        T_diff = T_diff.to(device)
        with autocast(device_type = device.type):
            output = model(X)
        loss += loss_fn(output, target, T_diff).item()
        prob = F.softmax(output, dim = 1)[..., 1] 
        scores.append(prob.unsqueeze(0).cpu())
        truths.append(target.long().unsqueeze(0).cpu())
        frame_ids.extend(list(fid))
print(loss/len(Loader))
def plot_likelihood(preds:torch.Tensor, truth: torch.Tensor)-> None:
    bins = [0.05 * i for i in range(21)]
    plt.xlim(0, 1) 
    plt.figure(figsize=(10, 6))
    plt.hist(preds[truth == 1], bins=bins, alpha=0.2, edgecolor='black', label = 'accident')
    plt.hist(preds[truth == 0], bins=bins, alpha=0.2, color='red', edgecolor='black', label = 'non-accident')
    plt.title('Model output')
    plt.xlabel('Output probability')
    plt.ylabel('event')
    plt.grid(alpha=0.3)
    plt.show()
    plt.legend()

    output = os.path.join(f'likelihood-weight_update-epoch4.png')
    plt.savefig(output)
    print('Check -> ', output)
    plt.savefig(output.replace('.png', '.pdf'))
    print('Check -> ', output.replace('.png', '.pdf')) 
    return 
def plot_likelihood_0p3(preds:torch.Tensor, truth: torch.Tensor)-> None:
    bins = [0.01 * i for i in range(31)]
    plt.xlim(0, 0.3) 
    plt.figure(figsize=(10, 6))
    plt.hist(preds[truth == 1], bins=bins, alpha=0.2, edgecolor='black', label = 'accident')
    plt.hist(preds[truth == 0], bins=bins, alpha=0.2, color='red', edgecolor='black', label = 'non-accident')
    plt.title('Model output')
    plt.xlabel('Output probability')
    plt.ylabel('event')
    plt.grid(alpha=0.3)
    #plt.show()
    plt.yscale('log')
    plt.legend()

    output = os.path.join(f'likelihood-weight_update0p3-epoch4.png')
    plt.savefig(output)
    print('Check -> ', output)
    plt.savefig(output.replace('.png', '.pdf'))
    print('Check -> ', output.replace('.png', '.pdf')) 
    return 
#scores = np.code_info([])
scores =  np.concatenate([p.flatten().cpu().numpy() for p in scores])
truths = np.concatenate([p.flatten().cpu().numpy() for p in truths])
fids = np.array(frame_ids)
plot_likelihood(preds = scores, truth = truths)
plot_likelihood_0p3(preds = scores, truth = truths)
def update_weight(metadata: pd.DataFrame, fids: np.ndarray, scores: np.ndarray, targets: np.ndarray, high_confidence_threshold: float = 0.95, low_confidence_threshold: float = 0.3) -> pd.DataFrame:
    metadata = metadata.copy()
    metadata['score'] = scores
    high_confidence_indices = np.where( ((targets == 0) & (scores <= (1 - high_confidence_threshold))) )[0]
    
    mask = metadata.apply(lambda row: (row.key in fids[high_confidence_indices]), axis=1)
    metadata.loc[mask, 'weight'] = 0
    
    low_confidence_indices = np.where( ((targets == 0) & (scores >= (1 - low_confidence_threshold))) | ((targets == 1) & (scores <= (low_confidence_threshold))) )[0]
    #print(scores[high_confidence_indices])
    #print(vids[targets == 0])
    #print(vids[high_confidence_indices])
    mask = metadata.apply(lambda row: (row.key in fids[low_confidence_indices]), axis=1)
    mask_pos = mask & (metadata.target == 1)
    mask_neg = mask & (metadata.target == 0)
    metadata.loc[mask_pos, 'weight'] *= 2
    #metadata.loc[mask_neg, 'weight'] *= 10
    metadata.to_csv('random_sampling-monitor/test.csv', index = False)
update_weight(metadata = dataset.metadata, fids = fids, scores = scores, targets = truths)
