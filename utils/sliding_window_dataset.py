import torch
from torch.utils.data import Dataset
import cv2
import pandas as pd
from torchvision import transforms
import numpy as np
import math
import os
import random
from typing import List, Tuple, Dict, Optional, Union
import torch.nn.functional as F
from scipy.special import expit  # sigmoid function

class SlidingWindowDataset(Dataset):
    """
    Dataset for loading video files with sliding window sampling and sigmoid probability targets.

    This dataset:
    1. Samples a fixed number of frames from each video at constant intervals
    2. Creates sliding windows of consecutive frames
    3. Assigns a target probability to each window based on a sigmoid function
    """
    def __init__(
        self,
        root_dir: str = "./dataset/train",
        csv_file: str = './dataset/train.csv',
        window_size: int = 16,
        total_frames: int = 32,
        augmentation_config: Optional[Dict[str, bool]] = None,
        global_augment_prob: float = 0.25,
        horizontal_flip_prob: float = 0.5,
        sigmoid_steepness: float = 5.0,
        training_mode: bool = True
    ):
        """
        Initialize the sliding window dataset.

        Args:
            root_dir (str): Directory containing the video files
            csv_file (str): Path to CSV file with video metadata
            window_size (int): Number of frames in each sliding window
            total_frames (int): Total number of frames to sample from each video
            augmentation_config (dict): Booleans to enable specific augmentations
            global_augment_prob (float): Probability of applying augmentation to a video
            horizontal_flip_prob (float): Probability of flipping a video horizontally
            sigmoid_steepness (float): Controls steepness of the sigmoid probability curve
            training_mode (bool): If True, uses sliding windows; if False, uses a single window per video
        """
        self.window_size = window_size
        self.total_frames = total_frames
        self.sigmoid_steepness = sigmoid_steepness
        self.training_mode = training_mode

        # Default augmentation configuration
        self.aug_config = {
            'fog': False,
            'noise': False,
            'gaussian_blur': False,
            'color_jitter': False,
            'horizontal_flip': False,
            'rain_effect': False,
        }

        # Override with user configuration if provided
        if augmentation_config:
            self._validate_config(augmentation_config)
            for key, value in augmentation_config.items():
                if key in self.aug_config:
                    self.aug_config[key] = value

        self.global_augment_prob = global_augment_prob
        self.horizontal_flip_prob = horizontal_flip_prob
        self.root_dir = root_dir
        self.data_frame = pd.read_csv(csv_file)

        # Basic transforms that are always applied
        self.base_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((112, 112)),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                std=[0.22803, 0.22145, 0.216989])
        ])

        # Prepare data
        if training_mode:
            # For training: create all possible windows for all videos
            self.video_windows = []  # Will contain (video_path, window_indices, target_probability)
            self._prepare_sliding_windows()
        else:
            # For inference/validation: one window per video
            self.video_indices = self.data_frame['id'].to_list()
            self.video_files = dict()
            global_index = 0
            for Index in self.video_indices:
                file = os.path.join(root_dir, f'{Index:05d}.mp4')
                if os.path.isfile(file):
                    data = self.data_frame[self.data_frame['id'] == Index]
                    target = data['target'].item()
                    time_of_alert = data['time_of_alert'].item() if target==1 else None
                    time_of_event = data['time_of_event'].item() if target==1 else None
                    self.video_files[global_index] = (
                        file, target, time_of_alert, time_of_event
                    )
                    global_index += 1

        if self.training_mode and not self.video_windows:
            raise RuntimeError(f"No valid video windows found in {root_dir}")
        elif not self.training_mode and not self.video_files:
            raise RuntimeError(f"No MP4 files found in {root_dir}")

    def _prepare_sliding_windows(self):
        """
        Pre-compute all possible sliding windows and their target probabilities
        for each video in the dataset.
        """
        for index, row in self.data_frame.iterrows():
            file = os.path.join(self.root_dir, f'{row["id"]:05d}.mp4')
            if os.path.isfile(file):
                target = row['target']
                time_of_alert = row['time_of_alert'] if target == 1 else None
                time_of_event = row['time_of_event'] if target == 1 else None

                # Get video properties
                video = cv2.VideoCapture(file)
                total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = video.get(cv2.CAP_PROP_FPS)
                fps = fps if 1 <= fps <= 120 else 30  # prevent abnormal fps values
                video.release()

                # Generate all frame indices using constant interval sampling
                sampled_indices = self._constant_interval_sampling(total_frames, fps, time_of_event, time_of_alert)

                # If this is a non-accident video (target=0)
                if target == 0:
                    # Create sliding windows with all probabilities set to 0
                    for i in range(len(sampled_indices) - self.window_size + 1):
                        window = sampled_indices[i:i+self.window_size]
                        self.video_windows.append((file, window, 0.0))
                else:
                    # For accident videos, calculate target probability for each window
                    for i in range(len(sampled_indices) - self.window_size + 1):
                        window = sampled_indices[i:i+self.window_size]
                        last_frame_idx = window[-1]

                        # Convert to time before accident
                        last_frame_time = last_frame_idx / fps
                        time_to_accident = time_of_event - last_frame_time
                        alert_to_event_time = time_of_event - time_of_alert

                        # Calculate probability using sigmoid function
                        prob = self._sigmoid_accident_probability(time_to_accident, alert_to_event_time)
                        self.video_windows.append((file, window, prob))

    def _sigmoid_accident_probability(self, time_to_accident, alert_to_event_time):
        """
        Generate a sigmoid probability curve for accident prediction.

        Args:
            time_to_accident (float): Time to accident in seconds (negative = after accident)
            alert_to_event_time (float): Time between alert and event in seconds

        Returns:
            float: Probability value between 0 and 1
        """
        if time_to_accident <= 0:  # After or at accident time
            return 1.0

        # Normalize time relative to the alert-to-event interval
        normalized_time = (time_to_accident - alert_to_event_time) / alert_to_event_time

        # Apply sigmoid transformation
        # We ensure it passes through (0,1) and (alert_time,0.5)
        return expit(-self.sigmoid_steepness * normalized_time)

    def _constant_interval_sampling(self, total_frames, fps, time_of_event=None, time_of_alert=None, video_duration_sec=10):
        """
        Sample frames at constant intervals, ensuring coverage of the relevant video portion.

        Args:
            total_frames (int): Total number of frames in the video
            fps (float): Frames per second
            time_of_event (float, optional): Time of accident event in seconds
            time_of_alert (float, optional): Time of alert in seconds
            video_duration_sec (float): Expected video duration to sample

        Returns:
            list: Frame indices to sample
        """
        # Handle non-accident videos or invalid timestamps
        if time_of_event is None or time_of_alert is None or time_of_event <= 0 or time_of_alert <= 0:
            # For non-accident videos, sample from a 10-second window
            frames_to_sample = min(total_frames, int(fps * video_duration_sec))
            interval = max(1, frames_to_sample // self.total_frames)
            indices = list(range(0, frames_to_sample, interval))[:self.total_frames]

            # If we don't have enough frames, duplicate the last ones
            while len(indices) < self.total_frames:
                indices.append(min(total_frames - 1, indices[-1] + 1))

            return indices

        # For accident videos, focus on the frames before the accident
        event_frame = min(total_frames - 1, int(time_of_event * fps))

        # Set the valid range for sampling (considering 10-second videos)
        start_frame = max(0, event_frame - int(video_duration_sec * fps))
        end_frame = event_frame - 1  # Exclude the event frame itself

        valid_frame_count = end_frame - start_frame + 1

        # Calculate sampling interval
        interval = max(1, valid_frame_count // self.total_frames)

        # Sample frames at constant intervals
        indices = list(range(start_frame, end_frame + 1, interval))[:self.total_frames]

        # Ensure we have enough frames
        while len(indices) < self.total_frames:
            # Add intermediate frames if possible
            if len(indices) >= 2:
                new_indices = []
                for i in range(len(indices) - 1):
                    new_indices.append(indices[i])
                    mid_point = (indices[i] + indices[i+1]) // 2
                    if mid_point not in indices and mid_point != indices[i] and mid_point != indices[i+1]:
                        new_indices.append(mid_point)
                new_indices.append(indices[-1])
                indices = new_indices
                if len(indices) >= self.total_frames:
                    indices = indices[:self.total_frames]
                    break
            else:
                # If can't add intermediate frames, duplicate the last one
                indices.append(min(total_frames - 1, indices[-1] + 1))

        return indices

    def _validate_config(self, config: Dict[str, bool]) -> None:
        """
        Validate the augmentation configuration dictionary.
        """
        valid_keys = set(self.aug_config.keys())
        for key, value in config.items():
            if key not in valid_keys:
                raise ValueError(f"Unknown augmentation type: {key}. "
                                f"Valid options are: {', '.join(valid_keys)}")
            if not isinstance(value, bool):
                raise ValueError(f"Augmentation config values must be boolean, got {type(value)} for {key}")

    def _add_noise(self, img: torch.Tensor, noise_factor: float = 0.1) -> torch.Tensor:
        """Add random noise to an image tensor."""
        noise = torch.randn_like(img) * noise_factor
        noisy_img = img + noise
        return torch.clamp(noisy_img, 0., 1.)

    def _add_fog(self, img: torch.Tensor, fog_intensity: float = 0.3) -> torch.Tensor:
        """Simulate fog effect by adding a bright overlay with reduced contrast."""
        fog = torch.ones_like(img) * fog_intensity
        foggy_img = img * (1 - fog_intensity) + fog
        return torch.clamp(foggy_img, 0., 1.)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if self.training_mode:
            return len(self.video_windows)
        else:
            return len(self.video_files)

    def __load_frames(self, video_path: str, frame_indices: List[int]) -> List[torch.Tensor]:
        """
        Load specified frames from a video file.

        Args:
            video_path (str): Path to the video file
            frame_indices (List[int]): Indices of frames to load

        Returns:
            List[torch.Tensor]: List of preprocessed frame tensors
        """
        video = cv2.VideoCapture(video_path)
        frames = []

        # Step 1: Decide whether to augment this video at all
        apply_augmentation = random.random() < self.global_augment_prob

        # Check if any augmentations are enabled
        any_augmentations_enabled = any([self.aug_config[key] for key in self.aug_config
                                        if key != 'horizontal_flip'])

        # Only apply augmentation if both conditions are met
        apply_augmentation = apply_augmentation and any_augmentations_enabled

        # Horizontal flip is handled separately
        apply_horizontal_flip = self.aug_config['horizontal_flip'] and random.random() < self.horizontal_flip_prob

        # Pre-configure transformations for the entire video to ensure consistency
        video_transforms = []

        # If we decided to augment, set up the augmentations
        if apply_augmentation:
            # Determine which specific effects to apply based on config
            apply_fog = self.aug_config['fog']
            apply_noise = self.aug_config['noise']
            apply_gaussian_blur = self.aug_config['gaussian_blur']
            apply_color_jitter = self.aug_config['color_jitter']

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
                hue_range = (0.0, 0.1)
                color_jitter = transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue_range
                )
                video_transforms.append(color_jitter)

            # For fog effect, pre-generate parameters
            if apply_fog:
                fog_intensity = random.uniform(0.1, 0.2)

            # For noise effect, pre-generate parameters
            if apply_noise:
                noise_factor = random.uniform(0.01, 0.03)

        # Load the requested frames
        for frame_idx in frame_indices:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = video.read()
            if not success:
                # Create a black frame if reading fails
                frame = np.zeros((112, 112, 3), dtype=np.uint8)

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply base transforms
            frame_tensor = self.base_transforms(frame.astype(np.float32) / 255.0)

            # Apply transformations if we're augmenting
            if apply_augmentation:
                # Apply torchvision transforms
                for transform in video_transforms:
                    frame_tensor = transform(frame_tensor)

                # Apply custom effects
                if apply_fog:
                    frame_tensor = self._add_fog(frame_tensor, fog_intensity=fog_intensity)

                if apply_noise:
                    frame_tensor = self._add_noise(frame_tensor, noise_factor=noise_factor)

            # Apply horizontal flip separately
            if apply_horizontal_flip:
                frame_tensor = torch.flip(frame_tensor, dims=[2])

            frames.append(frame_tensor)

        video.release()
        return frames

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a video clip and its target probability.

        Args:
            idx (int): Index of sample

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Video tensor of shape (channels, n_frames, height, width)
                - Target probability tensor
        """
        if self.training_mode:
            # Training mode: get pre-computed window
            video_path, frame_indices, target_prob = self.video_windows[idx]
            frames = self.__load_frames(video_path, frame_indices)
            return torch.stack(frames, dim=1), torch.tensor([target_prob], dtype=torch.float32)
        else:
            # Inference/validation mode: process whole video
            video_path, target, time_of_alert, time_of_event = self.video_files[idx]

            # Get video properties
            video = cv2.VideoCapture(video_path)
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video.get(cv2.CAP_PROP_FPS)
            fps = fps if 1 <= fps <= 120 else 30  # prevent abnormal fps values
            video.release()

            # Sample frames
            frame_indices = self._constant_interval_sampling(total_frames, fps, time_of_event, time_of_alert)

            # Load frames
            frames = self.__load_frames(video_path, frame_indices[:self.window_size])

            # For validation mode, we can use a single frame window from the start
            # If it's an accident video, calculate the probability, otherwise it's 0
            if target == 1 and time_of_event is not None and time_of_alert is not None:
                last_frame_idx = frame_indices[self.window_size-1]
                last_frame_time = last_frame_idx / fps
                time_to_accident = time_of_event - last_frame_time
                alert_to_event_time = time_of_event - time_of_alert
                prob = self._sigmoid_accident_probability(time_to_accident, alert_to_event_time)
            else:
                prob = 0.0

            return torch.stack(frames, dim=1), torch.tensor([prob], dtype=torch.float32)
