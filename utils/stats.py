import torch
import torch.nn
from typing import Tuple
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math

def case_counting(mode: str, output: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    if mode == 'volume':
        """
        Args: 
        mode (str): volumn and sequence
        output (torch.Tensor): (batch_size, n_classes) or (batch_size, n_frames, n_classes)
        target (torch.Tensor): (batch_size)
        """
        positive_probs = F.softmax(output, dim = -1)[:, 1]
        positive_cases = positive_probs >= 0.5
        positive_cases.requires_grad = False 
    elif mode == 'sequence':
        probs = F.softmax(output, dim = -1)
        positive_cases = torch.any(probs[:, :, 1] >= 0.5, dim = 1) # (batch_size)
    else:
        raise ValueError(f'No such mode: {mode}')
    negative_cases = ~positive_cases
    true_cases = target == 1 
    false_cases = ~true_cases
    
    return positive_cases, negative_cases, true_cases, false_cases



def eval_metrics(outputs: torch.Tensor, targets: torch.Tensor, fps=20.0):
    """
    :param: all_pred (N x T x 2), where N is number of videos, T is the number of frames for each video
    :param: all_labels (N,)
    :param: time_of_accidents (N,) int element
    :output: AP (average precision, AUC), mTTA (mean Time-to-Accident), TTA@R80 (TTA at Recall=80%)
    """
    N = target.shape[0]
    
    time_of_accidents = np.array(torch.full((N, ), 90))
    # 90th frame is starting frame of accident.
    
    outputs = np.array(outputs) 
    targets = np.array(targets)
    preds_eval = []
    min_pred = np.inf
    n_frames = 0
    for idx, toa in enumerate(time_of_accidents):
        if targets[idx] > 0:
            pred = outputs[idx, :int(toa), 1]  # positive video
        else:
            pred = outputs[idx, :, 0]  # negative video
        # find the minimum prediction
        min_pred = pred.min().item() if min_pred > pred.min().item() else min_pred
        preds_eval.append(pred)
        n_frames += len(pred)
    total_seconds = outputs.shape[1] / fps

    # iterate a set of thresholds from the minimum predictions
    # temp_shape = int((1.0 - max(min_pred, 0)) / 0.001 + 0.5) 
    Precision = np.zeros((1000))
    Recall = np.zeros((1000))
    Time = np.zeros((1000))
    cnt = 0
    for Th in np.arange(max(min_pred, 0), 1.0, 0.001):
        Tp = 0.0
        Tp_Fp = 0.0
        time = 0.0
        counter = 0.0  # number of TP videos
        # iterate each video sample
        for i in range(len(preds_eval)):
            # true positive frames: (pred->1) * (gt->1)
            tp =  np.where(preds_eval[i]*targets[i]>=Th)
            Tp += float(len(tp[0])>0)
            if float(len(tp[0])>0) > 0:
                # if at least one TP, compute the relative (1 - rTTA)
                time += tp[0][0] / float(time_of_accidents[i])
                counter = counter+1
            # all positive frames
            Tp_Fp += float(len(np.where(preds_eval[i]>=Th)[0])>0)
        if Tp_Fp == 0:  # predictions of all videos are negative
            continue
        else:
            Precision[cnt] = Tp/Tp_Fp
        if np.sum(targets) ==0: # gt of all videos are negative
            continue
        else:
            Recall[cnt] = Tp/np.sum(targets)
        if counter == 0:
            continue
        else:
            Time[cnt] = (1-time/counter)
        cnt += 1
    # sort the metrics with recall (ascending)
    new_index = np.argsort(Recall)
    Precision = Precision[new_index]
    Recall = Recall[new_index]
    Time = Time[new_index]
    # unique the recall, and fetch corresponding precisions and TTAs
    _,rep_index = np.unique(Recall,return_index=1)
    rep_index = rep_index[1:]
    new_Time = np.zeros(len(rep_index))
    new_Precision = np.zeros(len(rep_index))
    for i in range(len(rep_index)-1):
         new_Time[i] = np.max(Time[rep_index[i]:rep_index[i+1]])
         new_Precision[i] = np.max(Precision[rep_index[i]:rep_index[i+1]])
    # sort by descending order
    new_Time[-1] = Time[rep_index[-1]]
    new_Precision[-1] = Precision[rep_index[-1]]
    new_Recall = Recall[rep_index]
    # compute AP (area under P-R curve)
    AP = 0.0
    if new_Recall[0] != 0:
        AP += new_Precision[0]*(new_Recall[0]-0)
    for i in range(1,len(new_Precision)):
        AP += (new_Precision[i-1]+new_Precision[i])*(new_Recall[i]-new_Recall[i-1])/2

    # transform the relative mTTA to seconds
    mTTA = np.mean(new_Time) * total_seconds
    print("Average Precision= %.4f, mean Time to accident= %.4f"%(AP, mTTA))
    sort_time = new_Time[np.argsort(new_Recall)]
    sort_recall = np.sort(new_Recall)
    TTA_R80 = sort_time[np.argmin(np.abs(sort_recall-0.8))] * total_seconds
    print("Recall@80%, Time to accident= " +"{:.4}".format(TTA_R80))

    return AP, mTTA, TTA_R80


if __name__ ==  "__main__":
    
    mode = 'sequence'
    #output = torch.rand(size = (10, 2))
    #output = F.softmax(output, dim = 1)
    logits = torch.zeros((10, 100, 2))
    
    for k in range(10):
        prob = np.random.random() 
        print(f'prob: {prob}')
        if prob > 0.5:
            for i in range(100):
            # This creates values that gradually shift from favoring the first column to favoring the second column
                logits[k, i, 0] = 5 - i/10  # Values decrease from 5 to -5
                logits[k, i, 1] = -5 + i/10  # Values increase from -5 to 5
        else:
            logits[k, :, 0] = 5
            #print(logits[k])
    output = torch.softmax(logits, dim = -1)
    target = torch.randint(low = 0, high = 2, size = (10, ))
    print(target) 
    print(output.shape) 
    positive_cases, negative_cases, true_cases, false_cases = case_counting(mode, output, target)
    print(positive_cases.sum(), negative_cases.sum(), true_cases.sum(), false_cases.sum())
    AP, mTTA, mTTA_R80 = eval_metrics(output, target, fps = 30)
    
    
