from typing import Union
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
from typing import Dict, Union
from collections import defaultdict

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
            'title': 'Anticipation Loss',
            'y_lim': (0., 0.1)
        }, 
        '1': {
            'name': 'mPrec',
            'title': 'Averaged Precision',
            'y_lim': (0.4, 0.8)
        },
        '2': {
            'name': 'mRecall',
            'title': 'Averaged Recall',
            'y_lim': (0.4, 1.1)
        },
        '3': {
            'name': 'mAcc',
            'title': 'Averaged Accuracy',
            'y_lim': (0.4, 0.8)
        }
    }
    
    def __init__(self, save_path:str, tag:str, resume = False) -> None:
        self.save_path = os.path.join(save_path, f'monitor_{tag}')
        self.nmetric = len(self.metrics.items())
        self.resume = resume
        self.state = dict()

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
            Y_train = self.state['train'][metric['name']]
            Y_evaltrain = self.state['validation'][metric['name']]
            Y_best_point = self.state['best_point'][metric['name']]
            X_best_point = self.state['best_point']['current_epoch'] 
            
            x = np.arange(1, len(Y_train) + 1)
            self.ax[index].plot(x, Y_train, label = 'train')
            self.ax[index].plot(x, Y_evaltrain, label = 'eval-train')
            
            self.ax[index].set_ylim(*metric['y_lim'])
            self.ax[index].set_title(metric['title'])
            
            self.ax[index].axvline(
                x = X_best_point, 
                c = 'grey',
                linestyle = 'dashdot',
                alpha = 0.645
            )
            self.ax[index].axhline(
                y = Y_best_point, 
                c = 'grey',
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
            self.ax[index].legend(fontsize = 'small')
        self.fig.savefig(self.save_path + '.png') 
        self.fig.savefig(self.save_path + '.pdf') 
        print(f'Check {self.save_path}.png')
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
                    self.state[dataset][key].append(value)
            
        
        self.__record()
        self.__plot()
