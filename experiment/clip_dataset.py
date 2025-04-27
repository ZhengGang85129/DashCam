import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict, Optional
from torchvision import transforms
import pandas as pd
import os 
import random
from PIL import Image

class TrainingClipDataset(Dataset):
    '''
    The sampling approach is the sliding window.
    '''
    def __init__(
        self,
        root_dir: str = 'dataset/clip_source/train',
        csv_file: str = 'dataset/frame_train.csv',
        num_frames: int = 16,
        frame_window: int = 16,
        resize_shape: Tuple[int, int] = (128, 171),
        crop_size: Tuple[int, int] = (112, 112),
        augmentation_config: Optional[Dict[str, bool]] = None,
        global_augment_prob: float = 0.25,
        horizontal_flip_prob: float = 0.5,
        inference: bool = False
    ):
        self.root_dir = root_dir
        self.metadata = pd.read_csv(csv_file,
            dtype = {"clip_id": str, "video_id": "int64", "target": "int64", "weight": "float64", "T_diff": "float64", "frame": "int64"})
        
        self.num_frames = num_frames
        self.frame_window = frame_window 
        
        self.transforms = transforms.Compose(
            [
                transforms.Resize(resize_shape), #ex: Resize the 720 x 1280 ->  resize_shape (128 x 171 by default)
                transforms.ToTensor(), # Convert (H, W, C) with range [0, 255] into (C, H, W) with range (0, 1)
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
        
        
        self.inference = inference 

    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            idx: index for the video.
        Reture:
            frames(T, C, H, W): Tensor containing video frames with dimensions representing batch samples, temporal sequence, color channels, and spatial resolution.
            target(scalar): whether the frames are from the video that contains the accident.
            T_diff(scalar): difference of selected frame to accident frame.
        """ 
        row = self.metadata.iloc[idx]
        folder = os.path.join(self.root_dir, f"video_{row.video_id:05d}")
        
        frame_paths = [
           os.path.join(folder, f"frame_{frame:05d}.jpg") for  frame in range(row.frame - self.num_frames + 1, row.frame + 1)
        ]
        frames = []
        for frame_path in frame_paths:
            try:
                img = Image.open(frame_path).convert("RGB")
                frames.append(img)
            except FileNotFoundError:  
                print(f"[ERROR] {frame_path} not found.")
                break
        
        
        augment_params = self._get_augmentation_params() 
        frames = [self._apply_augmentations(self.transforms(frame), augment_params, Idx) for Idx, frame in enumerate(frames) ] 
        
        if len(frames) < self.num_frames:
            print('[WARNING] not success')
            print(row)

        clips = torch.stack(frames, dim = 0)

        return clips, row.target, row.T_diff, row.key, row.concerned
         
    def _validate_config(self, config: Dict[str, bool]) -> None:
        """ Validate the augmentation configuration dictionary. """
        valid_keys = set(self.aug_config.keys())
        for key, value in config.items():
            if key not in valid_keys:
                raise ValueError(f"Unknown augmentation type: {key}. "
                                f"Valid options are: {', '.join(valid_keys)}")
            if not isinstance(value, bool):
                raise ValueError(f"Augmentation config values must be boolean, got {type(value)} for {key}")

    def _get_augmentation_params(self) -> Dict:
        """Generate all augmentation parameters at once for consistency."""
        # Determine if we apply augmentation at all
        apply_augmentation = random.random() < self.global_augment_prob and any(
            self.aug_config[key] for key in self.aug_config if key != 'horizontal_flip'
        )

        params = {
            'apply_augmentation': apply_augmentation,
            'apply_horizontal_flip': self.aug_config['horizontal_flip'] and random.random() < self.horizontal_flip_prob,
            'transforms': [],
            'fog_intensity': random.uniform(0.1, 0.2) if apply_augmentation and self.aug_config['fog'] else 0,
            'noise_factor': random.uniform(0.01, 0.03) if apply_augmentation and self.aug_config['noise'] else 0,
            'rain': {
                'enabled': apply_augmentation and self.aug_config['rain_effect'],
                'drop_length': random.randint(1, 5),
                'drop_count': random.randint(10, 40)
            }
        }

        # Add specific transforms
        if apply_augmentation:
            if self.aug_config['gaussian_blur']:
                params['transforms'].append(
                    transforms.GaussianBlur(kernel_size=3, sigma=random.uniform(0.1, 1.0))
                )

            if self.aug_config['color_jitter']:
                params['transforms'].append(
                    transforms.ColorJitter(
                        brightness=random.uniform(0.8, 1.2),
                        contrast=random.uniform(0.8, 1.2),
                        saturation=random.uniform(0.8, 1.2),
                        hue=(0.0, 0.1)
                    )
                )

        return params

    def _apply_augmentations(self, frame: torch.Tensor, params: Dict, frame_idx: int) -> torch.Tensor:
        """Apply all augmentations to a single frame."""
        if not params['apply_augmentation'] and not params['apply_horizontal_flip']:
            return frame

        # Create a copy to avoid modifying the original
        augmented = frame.clone()

        # Apply standard transforms
        if params['apply_augmentation']:
            for transform in params['transforms']:
                augmented = transform(augmented)

            # Apply fog effect
            if params['fog_intensity'] > 0:
                fog = torch.ones_like(augmented) * params['fog_intensity']
                augmented = augmented * (1 - params['fog_intensity']) + fog

            # Apply noise effect
            if params['noise_factor'] > 0:
                noise = torch.randn_like(augmented) * params['noise_factor']
                augmented = augmented + noise

            # Apply rain effect
            if params['rain']['enabled']:
                augmented = self._simulate_rain(
                    augmented,
                    drop_length=params['rain']['drop_length'],
                    drop_count=params['rain']['drop_count'],
                    seed=frame_idx
                )

        # Apply horizontal flip separately
        if params['apply_horizontal_flip']:
            augmented = torch.flip(augmented, dims=[2])

        return torch.clamp(augmented, 0., 1.)

    def _simulate_rain(self, img: torch.Tensor, drop_length: int = 20, drop_width: int = 1,
                      drop_count: int = 20, seed: Optional[int] = None) -> torch.Tensor:
        """Simulate rain by adding random streaks to an image."""
        c, h, w = img.shape
        rain_img = img.clone()

        # Set random seed if provided for consistency across frames
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



if __name__ == "__main__":
    from torch.utils.data import WeightedRandomSampler
    from torch.utils.data import DataLoader
    from tqdm import tqdm 
    dataset = TrainingClipDataset(
        root_dir = 'dataset/image/validation',
        csv_file = 'dataset/frame_validation.csv'
    )
    
    dataframe = pd.read_csv('dataset/frame_validation.csv')
    weights = dataframe.weight
    
    sampler = WeightedRandomSampler(weights = weights, num_samples = 1200, replacement = False) 
    loader = DataLoader(dataset, batch_size = 10, sampler = sampler, num_workers = 8)
    
    for data in tqdm(loader):
        clips, target, T_diff = data 
        #print(clips.shape) 
        #print(clips, target, T_diff)