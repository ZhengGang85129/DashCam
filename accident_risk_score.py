import torch
import torch.nn as nn
from model import DSA_RNN
import os
from typing import Union, List, Tuple
from tool import get_device
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import time
import pandas as pd
import argparse
from datetime import datetime
import torch.nn.functional as F
'''
usage: python3 accident_risk_score.py --tag <TAG> --model_ckpt <path to model checkpoint> --video_id <VIDEOID> --csv_file <path to csv file> --output_dir eval
'''


def parse_args():
    parser = argparse.ArgumentParser(description='TTA inference')

    parser.add_argument('--tag', type=str, help='tag for output file', required = True)
    parser.add_argument('--model_ckpt', type=str, default='model/best_model_ckpt.pt',
                        help='path to models checkpoints (default: model/best_model_ckpt.pt)')
    #parser.add_argument('--monitor_dir', type=str, default='train',
    #                    help='directory to save monitoring plots (default: ./train)')
    parser.add_argument('--video_id', type = int, help = 'Id of video in the csv file. One should make sure the video_id exists in the input csv file.', default = 0)
    parser.add_argument('--csv_file', type = str, help = 'csv file that contains the video Id information.', default = 'dataset/evaluation_videos.csv') 
    parser.add_argument('--output_dir', type = str, help = 'Folder which storing the output plot.', default = 'eval') 
    parser.add_argument('--full_video', help = 'infer on full video or clips', action = 'store_true') 
    
    args = parser.parse_args()
    #parser.print_help()
    return args



class AccidentPredictor:
    def __init__(self, state_path: str, device: Union[torch.device, None] = None, output_dir: str = 'eval', tag: str = ''):
        self.device = device
        self.output_dir = output_dir
        self.model = self.load_model(state_path).to(self.device)
        self.tag = tag
        self.model.eval()

    def load_model(self, state_path) -> Union[nn.Module, DSA_RNN]: 
        model = DSA_RNN(hidden_size = 512) #32 FIX ME
        if not os.path.isfile(state_path):
            raise FileNotFoundError(f"The file {state_path} does not exits,") 
        
        if self.device is None:
            saved_state = torch.load(state_path, map_location = 'cpu')
        else:
            saved_state = torch.load(state_path, map_location = self.device)
        
        if isinstance(saved_state, dict):
            if 'model_state_dict' in saved_state:
                model.load_state_dict(saved_state['model_state_dict'])
            else:
                model.load_state_dict(saved_state)
        else:
            model = saved_state
        
        return model

    def __plot_tta(self, scores_over_frames: torch.Tensor)->None:
        Path(self.output_dir).mkdir(parents = True, exist_ok=True)
        
        
        frame_index = np.array([idx for idx in range(scores_over_frames.shape[0])])
        scores_over_frames = scores_over_frames.cpu().numpy()   # type: ignore
        
            
         
        plt.figure(figsize=(12, 6))
        
        plt.plot(frame_index, scores_over_frames, 'b-', linewidth = 2)
        threshold = 0.5
        plt.axhline(y = threshold, color = 'r', linestyle = '--', label = 'Threshold')
        if self.has_accident and self.full_video:
            
            mask = scores_over_frames > threshold
            if np.any(mask):
                alert_frame_index = np.argmax(mask) 
                if alert_frame_index != -1: 
                    plt.scatter(alert_frame_index, scores_over_frames[alert_frame_index].item(), color = 'red', s = 50)
                print(f'Alert frame predicted by model: {alert_frame_index}')
                plt.axvline(x = int(self.accident_time * self.fps), color = 'y', linestyle = '--', label = 'time of event(frame)')
        
        plt.xlabel('Time (frame index)')
        plt.ylabel('Accident probability')
        plt.title('Time-to-Accident Prediction')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        # Get current time
        current_time = datetime.now()

        # Format as %m%d%h%min (month-day-hour-minute)
        formatted_time = current_time.strftime("%m%d%H%M")
        plt.savefig(os.path.join(self.output_dir, f"tta_visualization-{args.tag}_{formatted_time}.png"))
        print(f'Check-> {os.path.join(self.output_dir, f"tta_visualization-{args.tag}_{formatted_time}.png")}') 
        plt.close()
        
    
    def predict_on(self, index: int = 0, csv_file: str = './dataset/train.csv', full_video: bool = False)-> np.ndarray:
        self.full_video = full_video 
        start = time.time()
        dataframe = pd.read_csv(csv_file)
        video_path = f'dataset/train/extracted/{index:05d}.mp4' if not full_video else f'dataset/train/{index:05d}.mp4'
        
        if not os.path.isfile(video_path):
            raise RuntimeError(f'No such video path: {video_path}') 
        if dataframe['target'][dataframe['id'] == index].item(): 
            print(f'Video Id: {dataframe["id"][dataframe["id"] == index].item()}')
            print('This video contains accident.')
            self.has_accident = True
        else:
            print('Regular driving video.')
            self.has_accident = False 
        with torch.no_grad():
            frames = self.__load_video(video_path)
            frames = frames.to(self.device)
            frames = frames.unsqueeze(0)
            
            #frames = frames.view(-1, 100, -1, -1, -1)
            n_frames = frames.shape[1]
            #n_chunks = n_frames // 100 
            scores_over_frames = []
            
            stride = 100
            n_windows = n_frames//stride # FIX ME 
            print(n_frames, n_windows) 
            for chunk in range(n_windows):
                last_t_step = (chunk + 1) * stride
                last_t_step = last_t_step if last_t_step < n_frames else n_frames - 1 
                scores, _ = self.model(frames)
                scores_over_frames.append(F.softmax(scores.squeeze(0), dim = 1)[:, 1])
                #print(scores.shape)
                #if chunk == 3: break
            #print(scores_over_frames)
            scores_over_frames = torch.cat(scores_over_frames)
            #scores_over_frames = scores_over_frames.squeeze(0)
        if self.has_accident:
            self.accident_time = dataframe['time_of_event'][dataframe["id"] == index].item()
            print(f'Accident time: at {dataframe["time_of_event"][dataframe["id"] == index].item():.2f} sec ({int(self.accident_time * self.fps)})')
        
        print(f'Inference time: {time.time() - start:.2f} sec')
        self.__plot_tta( scores_over_frames=scores_over_frames)
        
        return scores_over_frames # type: ignore
    
    def __load_video(self, video_path: str, resize_shape: Tuple[int, int] = (224, 224)) -> torch.Tensor:
        
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize frame
            # Apply transforms
            frames.append(torch.from_numpy(frame.astype(np.float32)) / 255.0)
                
            frame_count += 1
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Ensure we have enough frames
        return (torch.stack(frames).permute(0, 3, 1, 2))
        

def main():
   global args
   args = parse_args() 
   predictor = AccidentPredictor(state_path = args.model_ckpt, device = get_device(), tag = args.tag, output_dir = args.output_dir)
   output_score = predictor.predict_on(index = args.video_id, csv_file = args.csv_file, full_video = args.full_video)
    

if __name__ == '__main__':
    
  main() 
