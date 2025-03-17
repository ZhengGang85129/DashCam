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
from utils.sampling_strategy import uniform_random_sampling, alert_focused_sampling

class AugmentedVideoDataset(Dataset):
    """
    Enhanced dataset for loading video files with configurable data augmentation techniques.

    This dataset uses a two-level probability system:
    1. First decide whether to augment the video at all (with probability 0.25)
    2. If yes, apply all specified augmentation types to the entire video

    Horizontal flip is handled differently because it preserves physical plausibility.
    """
    def __init__(
        self,
        root_dir: str = "./dataset/train",
        csv_file: str = './dataset/train.csv',
        sampling_mode: str = 'alert_focused',
        num_frames: int = 16,
        augmentation_config: Optional[Dict[str, bool]] = None,
        global_augment_prob: float = 0.25,
        horizontal_flip_prob: float = 0.5,
    ):
        """
        Initialize the augmented video dataset.

        Args:
            root_dir (str): Directory containing the video files
            csv_file (str): Path to CSV file with video metadata
            sampling_mode (str): Strategy to sample the video frames
                Supported keys: 'alert_focused', 'random'
            num_frames (int): Number of frames to sample from each video
            augmentation_config (dict): Booleans to enable specific augmentations
                Supported keys: 'fog', 'noise', 'gaussian_blur', 'color_jitter',
                                'horizontal_flip', 'rain_effect'
            global_augment_prob (float): Probability of applying augmentation to a video
            horizontal_flip_prob (float): Probability of flipping a video horizontally
                                        (applied independently of other augmentations)
        """
        # Default configuration (no augmentation)
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
        self.video_indices = self.data_frame['id'].to_list()
        self.video_files = dict()
        global_index = 0
        self.num_frames = num_frames
        self.sampling_mode = sampling_mode

        # Basic transforms that are always applied
        self.base_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((112, 112)),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                std=[0.22803, 0.22145, 0.216989])
        ])

        # Load videos from directory
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

        if not self.video_files:
            raise RuntimeError(f"No MP4 files found in {root_dir}")

    def _validate_config(self, config: Dict[str, bool]) -> None:
        """
        Validate the augmentation configuration dictionary.

        Args:
            config (Dict[str, bool]): Configuration dictionary to validate

        Raises:
            ValueError: If any keys are unknown or values are not boolean
        """
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

    def __len__(self) -> int:
        """
        Return the number of videos in the dataset.

        Returns:
            int: Dataset length
        """
        return len(self.video_files)

    def __load_video(self, video_path: str, time_of_alert: float, time_of_event: float) -> List[torch.Tensor]:
        """
        Load video file and return preprocessed frames.

        Args:
            video_path (str): Path to video file

        Returns:
            List[torch.Tensor]: List of preprocessed frames
        """
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        fps = fps if 1.<= fps <= 120. else 30. # prevent abnormal fps values
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate interval for frame sampling
        interval = math.floor(total_frames / self.num_frames)

        frames_saved = 0
        frames = []

        # Step 1: Decide whether to augment this video at all
        apply_augmentation = random.random() < self.global_augment_prob

        # Check if any augmentations are enabled
        any_augmentations_enabled = any([self.aug_config[key] for key in self.aug_config
                                        if key != 'horizontal_flip'])

        # Only apply augmentation if both conditions are met
        apply_augmentation = apply_augmentation and any_augmentations_enabled

        # Horizontal flip is handled separately - it's always considered as it maintains physical plausibility
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
        if self.sampling_mode == 'alert_focused':
            frame_indices = alert_focused_sampling(
                total_frames=total_frames,
                num_frames=self.num_frames,
                time_of_event=time_of_event,
                time_of_alert=time_of_alert,
                fps=fps
            )
        else:  # uniform sampling
            frame_indices = uniform_random_sampling(
                total_frames=total_frames,
                num_frames=self.num_frames,
                fps=fps
            )

        for i in frame_indices:
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, frame = video.read()
            if not success:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to tensor first using base transforms
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

                if apply_rain:
                    frame_tensor = simulate_rain(
                        frame_tensor,
                        drop_length=rain_drop_length,
                        drop_count=rain_drop_count,
                        seed=i  # Use frame index for consistent but varied rain
                    )

            # Apply horizontal flip separately
            if apply_horizontal_flip:
                frame_tensor = torch.flip(frame_tensor, dims=[2])

            frames.append(frame_tensor)
            frames_saved += 1
            if frames_saved >= self.num_frames:
                break

        video.release()
        return frames

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a video clip and its label from the dataset.

        Args:
            idx (int): Index of video file

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Video tensor of shape (channels, n_frames, height, width)
                - Target label tensor
        """
        video_path, target, time_of_alert, time_of_event = self.video_files[idx]
        frames = self.__load_video(video_path, time_of_alert, time_of_event)

        # Stack frames into a single tensor
        return torch.stack(frames, dim=1), torch.tensor(target)

def simulate_rain(img: torch.Tensor, drop_length: int = 20, drop_width: int = 1,
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
