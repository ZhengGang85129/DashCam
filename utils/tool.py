from typing import Union
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
from typing import Dict, Union
from collections import defaultdict
import random
import torch.backends.cudnn as cudnn
import torch

def get_device()->torch.device:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA is available. Using GPU.") 
    else:
        device = torch.device('cpu')
        print(f"Device: {device}")
    return device

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    cudnn.deterministic = True
    cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
    
class AverageMeter(object):
    """computes and stores the average and the current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.current_value = 0
        self.avg_value = 0
        self.sum = 0
        self.count = 0 
    
    def update(self, value:float, weight:Union[int, float] = 1):
        self.current_value = value
        self.sum += self.current_value * weight
        self.count += weight 
        self.avg_value  = self.sum / self.count


class Monitor(object):
    metrics = {
        '0': {
            'name': 'mLoss',
            'title': 'Loss',
            'y_lim': (0.0, 1.5)
        }, 
        '1': {
            'name': 'mPrec',
            'title': 'Averaged Precision(critical clips)',
            'y_lim': (0.4, 1.1)
        },
        '2': {
            'name': 'mRecall',
            'title': 'Averaged Recall(critical clips)',
            'y_lim': (0.4, 1.1)
        },
        '3': {
            'name': 'mAcc',
            'title': 'Averaged Accuracy(critical clips)',
            'y_lim': (0.4, 1.1)
        },
        '4': {
            'name': 'mAP',
            'title': 'mean Averaged Precision(critical clips)',
            'y_lim': (0.4, 1.1)
        },
        '5': {
            'name': 'mSpec',
            'title': 'Averaged Specifity(critical clips)',
            'y_lim': (0.4, 1.1)
        }
        
    }
    
    def __init__(self, save_path:str, tag:str, resume = False, iterations_per_epoch: int = 64) -> None:
        self.save_path = os.path.join(save_path, f'monitor_{tag}')
        self.nmetric = len(self.metrics.items())
        self.resume = resume
        self.state = dict()
        self.state['iterations_per_epoch'] = iterations_per_epoch
    def reset(self) -> None:
        # Calculate figure dimensions to maintain 4:3 ratio for each subplot
        width_inches = 10  # Total figure width - adjust as needed
        subplot_width = width_inches / 2  # For 2 columns
        subplot_height = subplot_width * 3/4  # Apply 4:3 ratio
        total_height = subplot_height * ((self.nmetric + 1) // 2)  # Calculate height based on number of rows

        self.fig, self.ax = plt.subplots(
            (self.nmetric + 1) // 2, 2,
            figsize=(width_inches, total_height),
            layout="constrained",
            gridspec_kw={'width_ratios': [1, 1]}
        )
        self.ax = self.ax.flatten()  # Flatten the 2D array to 1D for consistent indexing

    def __plot(self) -> None:
        self.reset() 
        for index, (_, metric) in enumerate(self.metrics.items()):
            
            if metric['name'] == 'mLoss':
                n_steps_per_update = self.state['iterations_per_epoch']
            else:
                n_steps_per_update = 1 # 
             
            Y_train = self.state['train'][metric['name']]
            Y_evaltrain = self.state['validation'][metric['name']]
            Y_best_point = self.state['best_point'][metric['name']]
            X_best_point = self.state['best_point']['current_epoch'] * n_steps_per_update 
            
            x = np.arange(1, len(Y_train) + 1).tolist()
            epochs = [epoch * n_steps_per_update for epoch in x] 
            
            if metric['name'] == 'mLoss':
                Loss_record = self.state['train']['Loss_record']
                n_iterations = np.arange(1, len(Loss_record)+1)
                self.ax[index].plot(n_iterations, Loss_record, label = 'temporal-weighted loss(train/iteration)')
                self.ax[index].plot([epoch - 0.5 for epoch in epochs], self.state['train']['meanLossRecord'], 'm-o',label = 'temporal-weighted loss(train/epoch)')

                self.ax[index].plot([epoch - 0.5 for epoch in epochs], self.state['validation']['meanLossRecord'], 'c-o',label = 'temporal-weighted loss(val/epoch)')
                ax_twiny = self.ax[index].twiny() 
                ax_twiny.xaxis.set_label_position('top')
                ax_twiny.xaxis.tick_top()
                for epoch in range(len(epochs) + 1):
                    ax_twiny.axvline(x=epoch, color='gray', linestyle='--', alpha=0.3)
                #ax_twiny.set_xticks([epoch for epoch in range(len(epochs))] + [len(epochs) + 1]) 
            self.ax[index].plot(epochs, Y_train, 'g-o',label = 'critical-clip Loss(train)')
            self.ax[index].plot(epochs, Y_evaltrain, 'r-o',label = 'critical-clip Loss(val)')
            self.ax[index].set_ylim(*metric['y_lim'])
            self.ax[index].set_title(metric['title'])
            
            self.ax[index].axvline(
                x = X_best_point, 
                c = 'violet',
                linestyle = 'dashdot',
                alpha = 0.645
            )
            self.ax[index].axhline(
                y = Y_best_point, 
                c = 'violet',
                linestyle = 'dashdot',
                alpha = 0.645
            )
            self.ax[index].annotate(
                f'({X_best_point}, {Y_best_point:.3f})',
                (X_best_point, Y_best_point),
                textcoords = "offset points",
                xytext = (0, 5),
                ha = 'center',
                fontsize = 8
            )
            self.ax[index].scatter(X_best_point, Y_best_point, c = 'red', label = 'best point', s = 10)
            self.ax[index].legend(fontsize = 'small', loc = 'upper left')
        self.fig.savefig(self.save_path + '.png') 
        self.fig.savefig(self.save_path + '.pdf') 
        print(f'Check {self.save_path}.png')
        print(f'Check {self.save_path}.pdf')
        print(f'Check {self.save_path} for numerical results over epochs.')
        return
    def __record(self) -> None:
        
        with open(self.save_path, 'w') as stream:
            yaml.safe_dump(self.state, stream)     
        
        return
    def update(self, metrics: Dict[str, Dict]) -> None:
        
        
        
        for dataset, dataset_metrics in metrics.items():
            if self.state.get(dataset, None) is None:
                self.state[dataset] = dict() 
            if dataset == 'best_point':
                for key, value in dataset_metrics.items():
                    self.state[dataset][key] = value
            else: 
                for key, value in dataset_metrics.items():
                    if self.state[dataset].get(key, None) is None:
                        self.state[dataset][key] = []
                    if key == 'Loss_record':
                        self.state[dataset][key].extend(value)
                    else:
                        self.state[dataset][key].append(value)
            
        
        self.__record()
        self.__plot()


if __name__ == '__main__':
    train_iterations = np.arange(1, 1201).tolist() 
    train_loss = np.random.rand(1200) + 0.2  # Random data for illustration
    train_loss.sort()  # Making it trend downward
    
    train_loss = train_loss[::-1].tolist()  # Flip to get decreasing trend
    epoch_loss = np.random.rand(12) * 1 + 0.1  # Random data
    epoch_loss.sort()  # Making it trend downward
    epoch_loss = epoch_loss[::-1].tolist()  # Flip to get decreasing trend
    train_accuracy = np.random.rand(12) * 0.8 + 0.2  # Random data
    train_accuracy.sort()  # Making it trend downward
    train_accuracy = train_accuracy[::].tolist()  # Flip to get decreasing trend
    
    
    
    # Validation loss per epoch
    val_epochs = np.arange(1, 11).tolist()   # Assuming 10 epochs
    val_loss = np.random.rand(12) * 1 + 0.2  # Random data
    val_loss.sort()  # Making it trend downward
    val_loss2 = np.random.rand(200) 
    val_loss2.sort()  # Making it trend downward
    val_loss = val_loss[::-1].tolist() + val_loss2[::-1].tolist() # Flip to get decreasing trend
    # Assuming each epoch contains 100 iterations (adjust based on your data)
    val_accuracy = np.random.rand(12) * 0.7 + 0.1  # Random data
    val_accuracy.sort()  # Making it trend downward
    val_accuracy = val_accuracy[::].tolist()  # Flip to get decreasing trend
    # Assuming each epoch contains 100 iterations (adjust based on your data)
    iterations_per_epoch = 100 
    A = Monitor(save_path = './', tag = '123', resume = False, iterations_per_epoch = iterations_per_epoch)
    
    for epoch in range(12): 
        metrics = {
            'train':{
            'mPrec': train_accuracy[epoch],
            'mRecall': train_accuracy[epoch],
            'mLoss': epoch_loss[epoch],
            'mAcc': train_accuracy[epoch],
            'Loss_record': train_loss[epoch * iterations_per_epoch: (epoch + 1) * iterations_per_epoch] 
            },
            'validation': {
            'mPrec': val_accuracy[epoch],
            'mRecall': val_accuracy[epoch],
            'mLoss': val_loss[epoch],
            'mAcc': val_accuracy[epoch] 
            },
            'best_point':{
                'mLoss': val_loss[epoch],
                'mPrec': 1 + val_accuracy[epoch],
                'mRecall': 1 + val_accuracy[epoch],
                'mAcc': 1 + val_accuracy[epoch],
                'current_epoch': 1 + epoch
            }
        }
    
        A.update(metrics = metrics) 