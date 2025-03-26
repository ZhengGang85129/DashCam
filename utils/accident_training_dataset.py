import torch
from torch.utils.data import Dataset
import cv2
from typing import List, Tuple
from torchvision import transforms
import pandas as pd
import os 
import math
import numpy as np
import random

class PreAccidentTrainingDataset(Dataset):
    '''
    The sampling approach is the sliding window.
    '''
    def __init__(
        self,
        root_dir: str = 'dataset/train/train_video',
        csv_file: str = './dataset/extracted_train.csv',
        num_frames: int = 16,
        frame_window: int = 16,
        interested_interval: int = 100,
        resize_shape: Tuple[int, int] = (128, 171),
        crop_size: Tuple[int, int] = (112, 112),
        augmentation_config: Optional[Dict[str, bool]] = None,
        global_augment_prob: float = 0.25,
        horizontal_flip_prob: float = 0.5
    ):
        self.root_dir = root_dir
        self.data_frame = pd.read_csv(csv_file)
        
        self.video_indices = self.data_frame['id'].to_list()
        self.video_files = dict()
        
        global_index = 0
        self.num_frames = num_frames
        self.frame_window = frame_window
        self.interested_interval = interested_interval 
        
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(), # Convert (H, W, C) with range [0, 255] into (C, H, W) with range (0, 1)
                transforms.Resize(resize_shape), #ex: Resize the 720 x 1280 ->  resize_shape (128 x 171 by default)
                transforms.CenterCrop(crop_size),# crop the center of images with size: crop_size(112 x 112 by default)
                transforms.Normalize(
                    mean=[0.43216, 0.394666, 0.37645], 
                    std=[0.22803, 0.22145, 0.216989]), # normalize the values.
                
            ]
        )

        self.global_augment_prob = global_augment_prob
        self.horizontal_flip_prob = horizontal_flip_prob
        self.aug_config = {
            'fog': False,
            'noise': False,
            'gaussian_blur': False,
            'color_jitter': False,
            'horizontal_flip': False,
            'rain_effect': False,
        }

        # Use specified augmentation effects if provided
        if augmentation_config:
            self._validate_config(augmentation_config)
            for key, value in augmentation_config.items():
                if key in self.aug_config:
                    self.aug_config[key] = value
        
        for Index in self.video_indices:
            file = os.path.join(
                root_dir, f'{Index:05d}.mp4'
            )
            if os.path.isfile(file):
                start_frame = self.data_frame[self.data_frame['id'] == Index]['start_frame'].item()
                end_frame = self.data_frame[self.data_frame['id'] == Index]['end_frame'].item()
                if end_frame - start_frame + 1 < self.interested_interval + self.frame_window - 1:
                    continue  
                self.video_files[global_index] = (file, self.data_frame[self.data_frame['id'] == Index]['target'].item(), 
                self.data_frame[self.data_frame['id'] == Index]['start_frame'].item(), 
                self.data_frame[self.data_frame['id'] == Index]['end_frame'].item(), 
                self.data_frame[self.data_frame['id'] == Index]['event_frame'].item())
                global_index += 1 
        if not self.video_files:
            raise RuntimeError(f"No MP4 files found in {root_dir}")

    def __len__(self) -> int:
        return len(self.video_files)
    
    def __load_video(self, video_path: str, event_frame: int) -> Tuple[List[torch.Tensor], int]:
        
        # Calculate interval
        interval = math.floor(self.frame_window / self.num_frames) 
        
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
         
        # Step 1: Decide whether to augment this video at all
        apply_augmentation = random.random() < self.global_augment_prob
        any_augmentations_enabled = any([self.aug_config[key] for key in self.aug_config if key != 'horizontal_flip'])
        apply_augmentation = apply_augmentation and any_augmentations_enabled
        apply_horizontal_flip = self.aug_config['horizontal_flip'] and random.random() < self.horizontal_flip_prob
        video_transforms = []

        if apply_augmentation:
            apply_fog = self.aug_config['fog']
            apply_noise = self.aug_config['noise']
            apply_gaussian_blur = self.aug_config['gaussian_blur']
            apply_color_jitter = self.aug_config['color_jitter']
            apply_rain = self.aug_config['rain_effect']

            # Configure the effects that will be applied
            if apply_gaussian_blur:
                # Use same blur parameters for all frames
                sigma = random.uniform(0.1, 1.0)
                gaussian_blur = transforms.GaussianBlur(kernel_size=3, sigma=sigma)
                video_transforms.append(gaussian_blur)

            if apply_color_jitter:
                # Use same color parameters for all frames
                brightness = random.uniform(0.8, 1.2)
                contrast = random.uniform(0.8, 1.2)
                saturation = random.uniform(0.8, 1.2)
                hue_range = (0.0, 0.1)  # This is valid: range from 0 to 0.1 shift
                color_jitter = transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue_range
                )
                video_transforms.append(color_jitter)

            # For rain effect, pre-generate parameters
            if apply_rain:
                rain_drop_length = random.randint(1, 5)
                rain_drop_count = random.randint(10, 40)

            # For fog effect, pre-generate parameters
            if apply_fog:
                fog_intensity = random.uniform(0.1, 0.2)

            # For noise effect, pre-generate parameters
            if apply_noise:
                noise_factor = random.uniform(0.01, 0.03)

        
        # Load and process frames
        selected_frame = random.randint(0, self.interested_interval - 1) + (self.num_frames - 1) * interval # Randomly select a frame as the 'current frame'. 
        start_frame = selected_frame - (self.num_frames  - 1)* interval  # Indices falling within the range [start_frame, current_frame] represent past frames.
        frames_saved = 0
        frames = []
        
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        count = 0
        illed_frame = 0
        for Idx, frame_index in enumerate(range(start_frame , selected_frame + interval, interval)):
            success, frame = video.read()
            if not success:
                #print(frame_index)
                illed_frame = frame_index
                break
            count += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#shape: (720, 1280, 3)
            frame = self.transforms(frame) #shape: (3, 112, 112)

            # Apply transformations if we're augmenting
            if apply_augmentation:
                for transform in video_transforms:
                    frame_tensor = transform(frame_tensor)

                if apply_fog:
                    frame_tensor = self._add_fog(frame_tensor, fog_intensity=fog_intensity)

                if apply_noise:
                    frame_tensor = self._add_noise(frame_tensor, noise_factor=noise_factor)

                if apply_rain:
                    frame_tensor = self._simulate_rain(
                        frame_tensor,
                        drop_length=rain_drop_length,
                        drop_count=rain_drop_count,
                        seed=i  # Use frame index for consistent but varied rain
                    )

            # Apply horizontal flip separately
            if apply_horizontal_flip:
                frame_tensor = torch.flip(frame_tensor, dims=[2])

            frames_saved += 1
            frames.append(frame)
            if count > self.num_frames:
                raise ValueError
        
        if frames_saved < self.num_frames:
            print('not success')
            print(f'video_path: {video_path}')
            print(f'total saved frames: {frames_saved}. start frame: {selected_frame - (self.num_frames - 1) * interval}, end_frame: {selected_frame}, total_frame: {total_frames}, illed_frame: {illed_frame}')
            #raise RuntimeError(f'total saved frames: {frames_saved}. start frame: {selected_frame - (self.num_frames - 1) * interval}, end_frame: {selected_frame}, total_frame: {total_frames}, illed_frame: {illed_frame}')
                
        video.release()
        return frames, event_frame - selected_frame
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Args:
            idx: index for the video.
        Reture:
            frames(batch_size, num_frames, n_channels, height, width): Tensor containing video frames with dimensions representing batch samples, temporal sequence, color channels, and spatial resolution.
            target(batch_size): whether the frames are from the video that contains the accident.
            T_diff(batchsize): difference of selected frame to accident frame.
        """ 
        video_path, target, _, _, event_frame = self.video_files[idx]
        #print(video_path, target, start_frame, end_frame)
        frames, T_diff = self.__load_video(video_path, event_frame = event_frame)
        #print(T_diff)
        frames = torch.stack(frames, dim = 1).permute(1, 0, 2, 3)
        return frames, target, T_diff

    def _validate_config(self, config: Dict[str, bool]) -> None:
        """ Validate the augmentation configuration dictionary. """
        valid_keys = set(self.aug_config.keys())
        for key, value in config.items():
            if key not in valid_keys:
                raise ValueError(f"Unknown augmentation type: {key}. "
                                f"Valid options are: {', '.join(valid_keys)}")
            if not isinstance(value, bool):
                raise ValueError(f"Augmentation config values must be boolean, got {type(value)} for {key}")

    def _add_noise(self, img: torch.Tensor, noise_factor: float = 0.1) -> torch.Tensor:
        """
        Add random noise to an image tensor.

        Args:
            img (torch.Tensor): Input image tensor of shape [C, H, W]
            noise_factor (float): Intensity of the noise to add

        Returns:
            torch.Tensor: Noisy image tensor
        """
        noise = torch.randn_like(img) * noise_factor
        noisy_img = img + noise
        return torch.clamp(noisy_img, 0., 1.)

    def _add_fog(self, img: torch.Tensor, fog_intensity: float = 0.3) -> torch.Tensor:
        """
        Simulate fog effect by adding a bright overlay with reduced contrast.

        Args:
            img (torch.Tensor): Input image tensor of shape [C, H, W]
            fog_intensity (float): Intensity of the fog effect (0-1)

        Returns:
            torch.Tensor: Image tensor with fog effect
        """
        fog = torch.ones_like(img) * fog_intensity
        foggy_img = img * (1 - fog_intensity) + fog
        return torch.clamp(foggy_img, 0., 1.)

    def _simulate_rain(img: torch.Tensor, drop_length: int = 20, drop_width: int = 1,
                     drop_count: int = 20, seed: Optional[int] = None) -> torch.Tensor:
        """
        Simulate rain by adding random streaks to an image.

        Args:
            img (torch.Tensor): Input image tensor of shape [C, H, W]
            drop_length (int): Length of rain drops
            drop_width (int): Width of rain drops (default: 1)
            drop_count (int): Number of rain drops to add
            seed (int, optional): Random seed for reproducible rain patterns

        Returns:
            torch.Tensor: Image tensor with rain effect
        """
        c, h, w = img.shape
        rain_img = img.clone()

        # Set random seed if provided to ensure consistency across frames
        if seed is not None:
            random.seed(seed)

        for _ in range(drop_count):
            x = random.randint(0, w-1)
            y = random.randint(0, h-drop_length-1)

            # Add a white streak
            rain_value = torch.ones(c, drop_length, drop_width) * 0.8

            # Ensure we don't go out of bounds
            end_y = min(y + drop_length, h)
            end_x = min(x + drop_width, w)

            # Add the rain drop
            rain_img[:, y:end_y, x:end_x] = rain_img[:, y:end_y, x:end_x] * 0.2 + rain_value[:, :end_y-y, :end_x-x]

        # Reset random seed to avoid affecting other code
        if seed is not None:
            random.seed()

        return torch.clamp(rain_img, 0., 1.)

if __name__ == '__main__':
    train_dataset = PreAccidentTrainingDataset(
        root_dir = 'dataset/train/train_video',
        csv_file = './dataset/extracted_train.csv',
        num_frames = 16,
        frame_window = 16,
        interested_interval = 100,
        resize_shape = (128, 171),
        crop_size = (112, 112)
    )
    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = 16,
        shuffle = True,
        num_workers = 8,
        pin_memory = True
    )
    for data in data_loader:
        X, y, T = data
        #print(X.shape, y.shape, T.shape)
        #print(T)
