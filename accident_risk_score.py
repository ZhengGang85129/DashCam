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

class AccidentPredictor:
    def __init__(self, state_path: str, device: Union[torch.device, None] = None):
        self.device = device
        self.model = self.load_model(state_path).to(self.device)
        self.model.eval()

    def load_model(self, state_path) -> Union[nn.Module, DSA_RNN]: 
        model = DSA_RNN(hidden_size = 64) # FIX ME
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

    def __plot_tta(self, output_dir: str, scores_over_frames: torch.Tensor)->None:
        Path(output_dir).mkdir(parents = True, exist_ok=True)
        
        
        frame_index = np.array([idx for idx in range(scores_over_frames.shape[0])])
        scores_over_frames = scores_over_frames.cpu().numpy()   # type: ignore
        
            
         
        plt.figure(figsize=(12, 6))
        
        plt.plot(frame_index, scores_over_frames, 'b-', linewidth = 2)
        threshold = 0.5
        plt.axhline(y = threshold, color = 'r', linestyle = '--', label = 'Threshold')
        if self.has_accident:
            
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
        plt.savefig(os.path.join(output_dir, f"tta_visualization.png"))
        
        plt.close()
        
    
    def predict_on(self, index: int = 0, test_sample: bool = False, csv_file: str = './dataset/train.csv')-> np.ndarray:
        
        start = time.time()
        dataframe = pd.read_csv(csv_file)
        video_path = f'dataset/train/{index:05d}.mp4' if not test_sample else  f'dataset/test/{index:05d}.mp4' 
        
        if not os.path.isfile(video_path):
            raise RuntimeError(f'No such video path: {video_path}') 
        if test_sample:
            self.has_accident = True
        else: 
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
            scores_over_frames, _ = self.model(frames)
            scores_over_frames = scores_over_frames.squeeze(0)
        if self.has_accident:
            self.accident_time = dataframe['time_of_event'][dataframe["id"] == index].item()
            print(f'Accident time: at {dataframe["time_of_event"][dataframe["id"] == index].item():.2f} sec ({int(self.accident_time * self.fps)})')
        
        print(f'Inference time: {time.time() - start:.2f} sec')
        self.__plot_tta(output_dir = 'eval', scores_over_frames=scores_over_frames)
        
        return scores_over_frames
    
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
            frame = cv2.resize(frame, resize_shape)
            # Apply transforms
            frames.append(torch.from_numpy(frame))
                
            frame_count += 1
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # Ensure we have enough frames
        return (torch.stack(frames).permute(0, 3, 1, 2))
        

def main():
   predictor = AccidentPredictor(state_path = 'model/best_model_ckpt.pt', device = get_device())
   output_score = predictor.predict_on(index = 1388, test_sample = False, )
    

if __name__ == '__main__':
    
  main() 