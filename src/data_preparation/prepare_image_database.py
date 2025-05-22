import os, sys
from multiprocessing import Pool
sys.path.append(os.getcwd())
import pandas as pd
from typing import  Dict,  Optional, List
from functools import partial
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import random
import matplotlib.image as mpimg

class _Config:
    image_save_folder = 'dataset/img_database'
    input_dir = 'dataset'
    img_size = (320, 320)
    ROI = 60 # frame of positive videos within region of interested  (ROI) are classified as positive samples 

def prepare_parse(parser_name: str = 'prepare_dataset') -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(description = f'{parser_name} script with batch size argument')
    parser.add_argument('--dataset', type = str, choices = ['train', 'validation', 'test'], required = True)
    parser.add_argument('--metadata_path', type = str , required = True)
    parser.add_argument('--extract_frame', action = "store_true")
    args = parser.parse_args()
    return args

    
class Video_Processor:
    def __init__(self, metadata_path: str,  
                 dataset: str,
                 ):
        self.metadata = pd.read_csv(metadata_path)
        self.save_dir = Path(_Config.image_save_folder) / dataset     
        self.save_dir.mkdir(parents = True, exist_ok = True)
        self.input_dir = Path(_Config.input_dir)/ 'train' if dataset != 'test' else Path(_Config.input_dir) / 'test' 
        self.dataset = dataset
    @property
    def get_vid_list(self)->List[int]:
        return self.metadata.id.to_list()
    
    def process(self, video_id: int, extract_frame: bool = False) -> Optional[Dict]:
        '''
        Args:
            video_id(int): video id.
            extract_frame(bool): extract each frame as .jpg image from the video if True.
        Return:
            row(Optional): meta data information of each frame. columns: 
            - 'frame': frame index
            - 'video_id': video id
            - 'label': training label
            - 'positive': positive(1)/negative(0) samples 
            - 'fps': fps
            - 'weight': 1 (sample weight)
            - 'T_diff': difference of frame index to frame of event.
        '''
        
        video_path = Path(f'{self.input_dir}/{video_id:05d}.mp4')    
        assert video_path.exists(), f'No such mp4 file: {video_path}' 
        
        img_dir = Path(self.save_dir)/Path(f'video_{video_id:05d}')
        img_dir.mkdir(parents=False, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        target = self.metadata[self.metadata.id == video_id].target.item() if self.dataset != 'test' else None
        
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        time_of_event = self.metadata[self.metadata.id == video_id].time_of_event.item() if target == 1 else -1
        last_frame = int(time_of_event * fps) if target == 1 else int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
        
        
        frame_of_event = last_frame
        frame_idx = 0
        row = []
        
        while frame_idx < last_frame:
            ret, frame = cap.read()
            if not ret: break
            fpath = img_dir / Path(f'{frame_idx:05d}.jpg')
            ROI = True if (target == 1) and (frame_of_event - _Config.ROI <= frame_idx <= frame_of_event) else False # Region of interest
            
            if extract_frame:
                cv2.imwrite(fpath, cv2.resize(frame, _Config.img_size), [cv2.IMWRITE_JPEG_QUALITY, 90])
            if self.dataset != 'test':
                row.append(
                    {'frame': frame_idx,
                    'video_id': video_id,
                    'label': 1 if ROI else 0,  
                    'fps': fps,
                    'weight': 1.,
                    'T_diff': frame_of_event - frame_idx - 1 if ROI else -1,
                    'positive': 1 if target else 0,
                    'last_frame': last_frame - 1 
                    }
                )
            frame_idx += 1
        cap.release()
        return row

    def run(self, extract_frame: bool = False) -> None:
        video_list = self.get_vid_list
       
        partial_worker = partial(self.worker, process_fn = self.process, extract_frame = extract_frame)
        
        metadata_frame = []
        
        with Pool(processes = 8) as pool:
            for result in tqdm(pool.imap_unordered(partial_worker, video_list), total = len(video_list)):
                if self.dataset == 'test':
                    continue
                metadata_frame += result
        self.metadata_frame = pd.DataFrame.from_dict(metadata_frame)
        self.metadata_frame_path = Path(_Config.image_save_folder + f'/frame-metadata_{self.dataset}.csv')
        self.metadata_frame.to_csv(self.metadata_frame_path, index = False)
        
        print('Done')
        print(f'Check: {self.metadata_frame_path}.') 
        
        if extract_frame and self.dataset != 'test':
            self.visualize(n_samples = 3)
         
    def worker(self, video_id, process_fn, extract_frame: bool = False):
        return process_fn(video_id, extract_frame = extract_frame) 

    def visualize(self, n_samples = 3) -> None:
        #plt.figure(figsize = (16, 5))
        
        sampled_vid = random.sample(self.metadata_frame.video_id.unique().tolist(), n_samples)
        
        for idx, vid in enumerate(sampled_vid):
            fig, axes = plt.subplots(4, 4, figsize = (12, 12)) 
            axes = axes.flatten()
            img_dir = Path(self.save_dir)/Path(f'video_{vid:05d}')   
            image_paths = sorted(img_dir.glob("*.jpg"), reverse = True)[:16]
            
            for idx, ax in enumerate(axes):
                img = mpimg.imread(image_paths[idx])
                ax.imshow(img)
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(self.save_dir / Path(f'/vid_{vid}.png'))
            plt.close()
def main():
    args = prepare_parse() 
    print('Generating ', args.dataset, ' dataset.')
    print('Meta data: ',args.metadata_path) 
    
    processor = Video_Processor(
        metadata_path = args.metadata_path,
        dataset = args.dataset
        )

    processor.run(extract_frame = args.extract_frame)
if __name__ == "__main__":
    main()
