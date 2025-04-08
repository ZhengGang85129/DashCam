import pandas as pd
from typing import List, Tuple, Dict, Union, cast
import cv2
import os

def extract_frames_before_timestamp(output_dir: str = './', video_info: Dict = None, pre_accident_stamps = [500, 1000, 1500], n_frames:int = 16):
    """
    
    """
    # Create output directory if specified and doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    src_cap = video_info['cap']
    src_path = video_info['video_path']
    if not src_cap.isOpened():
        raise ValueError(f"Could not open video file: {src_path}")
    src_fps = video_info['fps']
    src_event_time = video_info['time_of_event'] if video_info['accident'] else video_info['duration']/2.
    src_Index = video_info['Index']
    src_width = video_info['width']
    src_height = video_info['height']
    #max_pretime = video_info['time_of_alert']
    #clip_paths = []
     
    for pre_time in pre_accident_stamps:
        
        if  src_event_time*1000 < pre_time: continue
        end_time = src_event_time - pre_time/1000  
        end_frame = int(end_time * src_fps) - 1
        start_frame = max(0, end_frame - int(n_frames - 1))
        output_path = os.path.join(
            output_dir, f"tta_{pre_time}ms/{src_Index:05d}.mp4"
        )
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        out = cv2.VideoWriter(output_path, fourcc, src_fps, (src_width, src_height))
        
        # Set frame position to start frame
        src_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Read and write frames
        print(end_frame, start_frame)
        for _ in range(start_frame, end_frame + 1):
            ret, frame = src_cap.read()
            if not ret:
                break
            out.write(frame)
        
        # Release writer
        out.release()
        #clip_paths.append(output_path)
        print(f"Created clip ending {pre_time}ms before accident: {output_path}") 
    src_cap.release()
    #return clip_paths

def pick(Index: int, dataframe: pd.DataFrame)->Tuple[str, float]:
    
    video_path = f'./dataset/train/{Index:05d}.mp4'
    
    video_info = dict()
    video_info.update(get_video_info(video_path = video_path))
    video_info['time_of_event'] = dataframe[dataframe['id'] == Index]["time_of_event"].item()
    video_info['time_of_alert'] = dataframe[dataframe['id'] == Index]["time_of_alert"].item()
    video_info['Index'] = Index
    video_info['accident'] = True if dataframe[dataframe['id'] == Index]["target"].item() > 0.5 else False

    return video_info




def get_video_info(video_path: str) -> Dict[str, Union[float, int, str, cv2.VideoCapture]]:
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return {
        'video_path': video_path,
        'frame_counts': total_frames,
        'fps': fps,
        'height': height,
        'width': width,
        'duration': total_frames/fps,
        'cap': cap 
    }


if __name__ == "__main__":
    for dataset  in [("train", "evaluation-train"), ("validation", "evaluation")]:
        validation_dataset_csv = f'./dataset/sliding_window/{dataset[0]}_videos.csv'
        
        dataframe = pd.read_csv(validation_dataset_csv)
        output_dir = f'dataset/sliding_window/{dataset[1]}'
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        pre_accident_stamps = [500, 1000, 1500]
        
        for pre_time in pre_accident_stamps:
            if not os.path.exists(os.path.join(output_dir, f'tta_{pre_time}ms')):
                os.makedirs(os.path.join(output_dir, f'tta_{pre_time}ms'))
        
        
        for _, row in dataframe.iterrows():
            extract_frames_before_timestamp(output_dir = output_dir, video_info = pick(int(row.id), dataframe = dataframe,), pre_accident_stamps = pre_accident_stamps, n_frames = 16) 
        
     
    
    
     