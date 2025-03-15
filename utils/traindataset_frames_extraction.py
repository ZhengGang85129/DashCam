import pandas as pd
from typing import List, Tuple, Dict, Union, cast
from functools import partial
import os
import cv2


def extract_clips(cap: cv2.VideoCapture, time_of_event:float, output_path:str, width:int, height: int, last_nframes: int = 10, nframes: int = 100, fps: float = 30.0)-> None:
    """
    Extract frames from a specific time interval
    
    Args:
        cap (str): Path to video file.
        output_path (str): path saves output video.
        width (int): width of video.
        height (int): height of video.
        time_of_event (int): Time of event occurs.
        last_nframes (int): number of the frames containing accident.
        nframes (int): total number of frames for the clip
        fps (float): frame rate. 
    Return:
        clips (decord.ndarray.NDArray): clips that used for training 
    """
    start_frame = int(time_of_event * fps) - nframes + last_nframes
    end_frame = int(time_of_event * fps) + last_nframes 
    
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Set starting position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= (end_frame - start_frame):
            break
            
        out.write(frame)
            
        frame_count += 1
    
    cap.release()
    out.release()
 

def pick(Index: int, dataframe: pd.DataFrame)-> Tuple[str, float]:

    video_path = f'./dataset/train/{Index:05d}.mp4' 
    time_of_event = dataframe[dataframe['id'] == Index]["time_of_alert"].item()
    
    return video_path, time_of_event 


     
def get_video_info(video_path: str) -> Dict[str, Union[float, int, str, cv2.VideoCapture]]:
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
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
    




def main():
    df = pd.read_csv('dataset/train.csv')
    
    clip =  partial(extract_clips, last_nframes = 0, nframes = 150) 
    output_dir = 'dataset/train/extracted'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)  
    for _, row in df.iterrows():
        #print(int(row.id), row.time_of_event)
        video_path, time_of_event = pick(Index = int(row.id), dataframe = df)
        video_prop = get_video_info(video_path = video_path)
        duration = cast(float, video_prop.get('duration'))
        #assert video_prop.get('fps') > 0 
        if video_prop.get('fps') is None:
            raise RuntimeError(f'fps is None value')
        if not isinstance(video_prop.get('fps'), (int, float)):
            raise TypeError(f"FPS must be a number, got {type(video_prop.get('fps')).__name__}")
        assert cast(float, video_prop.get('fps')) > 0, f"FPS must be positive, got {cast(float, video_prop.get('fps'))}"
        
        pivot_time = time_of_event if row.target else (duration/2.) 
     
        print(f'pivot time: {pivot_time:.3f} sec, target: {row.target}, total duration: {duration:3f}')
        try:
            output_path = os.path.join(output_dir, f'{int(row.id):05d}.mp4')
            print(f"Processing {int(row.id):05d}.mp4 ...")
            clip(cap = cast(cv2.VideoCapture, video_prop.get('cap')), time_of_event = cast(float, pivot_time), fps = cast(float, video_prop.get('fps')) , output_path = cast(str, output_path), width = cast(int, video_prop.get('width')), height = cast( int, video_prop.get('height')))
             
        except Exception as e:
            print(f"Error processing {int(row.id):05d}.mp4: {e}")
 
if __name__ == "__main__":
    main()
