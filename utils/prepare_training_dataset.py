import os, sys
sys.path.append(os.getcwd())
import pandas as pd

from typing import List, Tuple, Dict, Union, cast, Optional
from functools import partial

import cv2

from utils.misc import parse_args
import argparse


def parser() -> argparse.ArgumentParser:

    parser = parse_args(parser_name = 'preprocess')
    parser.add_argument('--dataset', type = str, choices = ['train', 'validation'])

    return parser.parse_args()



def extract_clips(cap: cv2.VideoCapture, time_of_event:float, time_of_alert: Optional[float], output_path:str, width:int, height: int,  fps: float = 30.0, pre_accident_frames: int = 105, post_accident_frames: int = 10)-> Tuple[int, int, int, int]:
    """
    Extract frames from a specific time interval
    
    Args:
        cap (str): Path to video file.
        output_path (str): path saves output video.
        width (int): width of video.
        height (int): height of video.
        time_of_event (int): Time of event occurs.
        nframes (int): total number of frames for the clip
        fps (float): frame rate. 
    Return:
        clips (decord.ndarray.NDArray): clips that used for training 
    """
    event_frame = int((time_of_event * fps) - 1)
    alert_frame = int((time_of_alert * fps) - 1) if time_of_alert is not None else None
    end_frame = event_frame + post_accident_frames -1 
    start_frame = max(0, event_frame - pre_accident_frames)
     
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Set starting position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    n_frames = 0 
    for _ in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        n_frames += 1
     
    
    cap.release()
    out.release()
    print(f'n_frames: {n_frames}')
    if alert_frame is None:
        return start_frame - start_frame, end_frame - start_frame, event_frame - start_frame, alert_frame 
        
    return start_frame - start_frame, end_frame - start_frame, event_frame - start_frame, alert_frame - start_frame  
 

def pick(Index: int, dataframe: pd.DataFrame)-> Tuple[str, float]:

    video_path = f'./dataset/train/{Index:05d}.mp4' 
    time_of_event = dataframe[dataframe['id'] == Index]["time_of_event"].item()
    time_of_alert = dataframe[dataframe['id'] == Index]["time_of_alert"].item() 
    return video_path, time_of_event, time_of_alert 


     
def get_video_info(video_path: str) -> Dict[str, Union[float, int, str, cv2.VideoCapture]]:
    print(video_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return {
        'frame_counts': total_frames,
        'fps': fps,
        'height': height,
        'width': width,
        'duration': total_frames/fps,
        'cap': cap 
    }
    


import sys

def main():
    global args
    args = parser()
    output_dir = f'dataset/sliding_window'

    df = pd.read_csv(f'{output_dir}/{args.dataset}_videos.csv')
    
    clip =  partial(extract_clips) 
    extracted_df = []
    
    os.makedirs(output_dir, exist_ok = True)
    os.makedirs(os.path.join(output_dir, args.dataset), exist_ok = True)  
    for _, row in df.iterrows():
        #print(int(row.id), row.time_of_event)
        video_path, time_of_event, time_of_alert = pick(Index = int(row.id), dataframe = df)
        video_prop = get_video_info(video_path = video_path)
        duration = cast(float, video_prop.get('duration'))
        #assert video_prop.get('fps') > 0 
        if video_prop.get('fps') is None:
            raise RuntimeError(f'fps is None value')
        if not isinstance(video_prop.get('fps'), (int, float)):
            raise TypeError(f"FPS must be a number, got {type(video_prop.get('fps')).__name__}")
        assert cast(float, video_prop.get('fps')) > 0, f"FPS must be positive, got {cast(float, video_prop.get('fps'))}"
        
        pivot_time = time_of_event if row.target else (duration/2.) 
        alert_time = time_of_alert if row.target else None 
     
        print(f'pivot time: {pivot_time:.3f} sec, target: {row.target}, total duration: {duration:3f}')
        try:
            output_path = os.path.join(output_dir, f'{args.dataset}/{int(row.id):05d}.mp4')
            print(f"Processing {int(row.id):05d}.mp4 ...")
            start_frame, end_frame, event_frame, alert_frame = clip(cap = cast(cv2.VideoCapture, video_prop.get('cap')), time_of_event = cast(float, pivot_time), fps = cast(float, video_prop.get('fps')) , output_path = cast(str, output_path), width = cast(int, video_prop.get('width')), height = cast( int, video_prop.get('height')), time_of_alert = alert_time)
            print(start_frame, end_frame, event_frame, alert_frame) 
            extracted_df.append(
                {'id': int(row.id),
                 'start_frame': start_frame,
                 'end_frame': end_frame,
                 'event_frame': event_frame,
                 'alert_frame': int(alert_frame) if alert_frame is not None else alert_frame,
                 'target': int(row.target)},
            )
        except Exception as e:
            print(f"Error processing {int(row.id):05d}.mp4: {e}")
    extracted_df = pd.DataFrame(extracted_df)
    extracted_df.to_csv(f'{output_dir}/{args.dataset}_extracted_clips.csv', index = False)
 
if __name__ == "__main__":
    main()
