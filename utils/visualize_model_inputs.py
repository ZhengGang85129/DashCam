#!/usr/bin/env python3
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import random
from torchvision import transforms
from pathlib import Path
import pandas as pd
from typing import List, Tuple, Optional

# Import directly from your module
from augmented_dataset import AugmentedVideoDataset, simulate_rain


def denormalize_tensor(tensor):
    """
    Denormalize a tensor that was normalized with the dataset's mean and std
    """
    mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1)
    return tensor * std + mean


def tensor_to_numpy_image(tensor):
    """
    Convert a torch tensor to a numpy image for display
    """
    if tensor.max() <= 1.0 and tensor.min() >= -1.0:
        # Denormalize if the tensor seems to be normalized
        if tensor.min() < 0:
            tensor = denormalize_tensor(tensor)

    # Ensure values are in [0, 1]
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to numpy and adjust dimensions for matplotlib
    return tensor.permute(1, 2, 0).numpy()


def get_smart_frame_indices(video_path: str, num_frames: int, time_of_event: Optional[float] = None) -> List[int]:
    """
    Generate smart frame indices that include the accident frame if time_of_event is provided

    Args:
        video_path: Path to the video file
        num_frames: Number of frames to sample
        time_of_event: Time in seconds when the accident occurs

    Returns:
        List of frame indices to sample
    """
    # Open the video to get metadata
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        return list(range(num_frames))  # Fallback

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    video_duration = total_frames / fps if fps > 0 else 0
    video.release()

    if total_frames <= num_frames:
        # If video has fewer frames than requested, return all frames
        return list(range(total_frames))

    if time_of_event is not None and fps > 0:
        # If we know when the accident happens, focus around that time
        event_frame = int(time_of_event * fps)

        # Define a window around the event
        # We'll take some frames before and after the event
        window_size = num_frames
        start_frame = max(0, event_frame - window_size//2)
        end_frame = min(total_frames, start_frame + window_size)

        # Adjust start if end is capped
        if end_frame == total_frames:
            start_frame = max(0, total_frames - window_size)

        # Generate indices within the window
        indices = np.linspace(start_frame, end_frame-1, num_frames, dtype=int)
        return indices.tolist()
    else:
        # Default: evenly spaced frames
        return np.linspace(0, total_frames-1, num_frames, dtype=int).tolist()


def add_fog(img, fog_intensity=0.15):
    """Simulate fog effect by adding a bright overlay with reduced contrast"""
    fog = torch.ones_like(img) * fog_intensity
    foggy_img = img * (1 - fog_intensity) + fog
    return torch.clamp(foggy_img, 0., 1.)


def add_noise(img, noise_factor=0.02):
    """Add random noise to an image tensor"""
    noise = torch.randn_like(img) * noise_factor
    noisy_img = img + noise
    return torch.clamp(noisy_img, 0., 1.)


def main():
    parser = argparse.ArgumentParser(description='Visualize augmentations from AugmentedVideoDataset')
    parser.add_argument('--csv_file', type=str, default='./dataset/train_videos.csv',
                       help='Path to CSV file with video metadata')
    parser.add_argument('--root_dir', type=str, default='./dataset/train/',
                       help='Directory containing video files')
    parser.add_argument('--output_dir', type=str, default='augmentation_visualizations',
                       help='Directory to save visualization results')
    parser.add_argument('--num_videos', type=int, default=3,
                       help='Number of videos to visualize')
    parser.add_argument('--frames_per_video', type=int, default=4,
                       help='Number of frames to show from each video')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--accident_only', action='store_true',
                       help='Only visualize videos with accidents (target=1)')
    parser.add_argument('--video_id', type=int, default=None,
                       help='Specific video ID to visualize (overrides num_videos)')
    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Check if the CSV file exists
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file {args.csv_file} not found.")
        return

    # Read the CSV to get available video IDs
    try:
        df = pd.read_csv(args.csv_file)

        # Filter videos based on command line arguments
        if args.video_id is not None:
            # Use specific video ID if provided
            if args.video_id in df['id'].values:
                selected_video_ids = [args.video_id]
            else:
                print(f"Error: Video ID {args.video_id} not found in CSV.")
                return
        else:
            # Filter for accident videos if requested
            if args.accident_only:
                filtered_df = df[df['target'] == 1]
                print(f"Found {len(filtered_df)} videos with accidents")
                if len(filtered_df) == 0:
                    print("No accident videos found. Try without --accident_only flag.")
                    return
                all_video_ids = filtered_df['id'].tolist()
            else:
                all_video_ids = df['id'].tolist()

            # Randomly select videos if there are enough
            if len(all_video_ids) > args.num_videos:
                selected_video_ids = random.sample(all_video_ids, args.num_videos)
            else:
                selected_video_ids = all_video_ids

        print(f"Selected videos for visualization: {selected_video_ids}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Create different augmentation configurations to compare
    augmentation_configs = [
        {
            "name": "No Augmentation",
            "config": None,
            "global_prob": 0.0
        },
        {
            "name": "Horizontal Flip Only",
            "config": {"horizontal_flip": True},
            "global_prob": 1.0,
            "horizontal_flip_prob": 1.0  # Always apply for visualization
        },
        {
            "name": "Fog Effect",
            "config": {"fog": True},
            "global_prob": 1.0
        },
        {
            "name": "Noise Effect",
            "config": {"noise": True},
            "global_prob": 1.0
        },
        {
            "name": "Gaussian Blur",
            "config": {"gaussian_blur": True},
            "global_prob": 1.0
        },
        {
            "name": "Color Jitter",
            "config": {"color_jitter": True},
            "global_prob": 1.0
        },
        {
            "name": "Rain Effect",
            "config": {"rain_effect": True},
            "global_prob": 1.0
        },
        {
            "name": "All Augmentations",
            "config": {
                "fog": True,
                "noise": True,
                "gaussian_blur": True,
                "color_jitter": True,
                "horizontal_flip": True,
                "rain_effect": True
            },
            "global_prob": 1.0
        }
    ]

    # Process each selected video
    for video_id in selected_video_ids:
        print(f"Processing video ID: {video_id}")

        # Create a specific CSV just for this video, preserving all original fields
        video_row = df[df['id'] == video_id]
        temp_csv_path = os.path.join(args.output_dir, f"temp_video_{video_id}.csv")
        video_row.to_csv(temp_csv_path, index=False)

        # Extract accident time information if available
        target = int(video_row['target'].values[0])
        time_of_event = None
        if target == 1 and 'time_of_event' in video_row.columns:
            if not pd.isna(video_row['time_of_event'].values[0]):
                time_of_event = float(video_row['time_of_event'].values[0])
                print(f"  Accident occurs at {time_of_event:.2f} seconds")

        # Create a figure to show all augmentations side by side
        fig, axes = plt.subplots(len(augmentation_configs), args.frames_per_video,
                                figsize=(4*args.frames_per_video, 3*len(augmentation_configs)))

        if len(augmentation_configs) == 1 and args.frames_per_video == 1:
            axes = np.array([[axes]])
        elif len(augmentation_configs) == 1:
            axes = axes.reshape(1, -1)
        elif args.frames_per_video == 1:
            axes = axes.reshape(-1, 1)

        plt.subplots_adjust(hspace=0.3, wspace=0.1)

        # Get the video path
        video_file = os.path.join(args.root_dir, f'{video_id:05d}.mp4')

        # Get smart frame indices based on accident time if available
        smart_indices = get_smart_frame_indices(
            video_path=video_file,
            num_frames=args.frames_per_video,
            time_of_event=time_of_event
        )

        # Process each augmentation configuration
        for config_idx, aug_config in enumerate(augmentation_configs):
            print(f"  Applying: {aug_config['name']}")

            # Configure the dataset with this augmentation
            horizontal_flip_prob = aug_config.get('horizontal_flip_prob', 0.5)

            # Create a function to process the frames with the given augmentation
            def visualize_with_custom_frames():
                # Open the video
                video = cv2.VideoCapture(video_file)
                if not video.isOpened():
                    raise ValueError(f"Could not open video file: {video_file}")

                # Apply augmentation decision
                apply_augmentation = aug_config["global_prob"] > 0 and aug_config["config"] is not None
                any_augmentations_enabled = False

                if apply_augmentation and aug_config["config"] is not None:
                    any_augmentations_enabled = any([aug_config["config"].get(key, False)
                                                   for key in ['fog', 'noise', 'gaussian_blur',
                                                              'color_jitter', 'rain_effect']])

                apply_augmentation = apply_augmentation and any_augmentations_enabled
                apply_horizontal_flip = False

                if aug_config["config"] is not None:
                    apply_horizontal_flip = aug_config["config"].get('horizontal_flip', False)
                    if apply_horizontal_flip:
                        apply_horizontal_flip = random.random() < horizontal_flip_prob

                # Set up video-level transforms
                video_transforms = []

                # Configure augmentations with fixed parameters for visualization
                if apply_augmentation:
                    apply_fog = aug_config["config"].get('fog', False) if aug_config["config"] else False
                    apply_noise = aug_config["config"].get('noise', False) if aug_config["config"] else False
                    apply_gaussian_blur = aug_config["config"].get('gaussian_blur', False) if aug_config["config"] else False
                    apply_color_jitter = aug_config["config"].get('color_jitter', False) if aug_config["config"] else False
                    apply_rain = aug_config["config"].get('rain_effect', False) if aug_config["config"] else False

                    if apply_gaussian_blur:
                        gaussian_blur = transforms.GaussianBlur(kernel_size=3, sigma=0.5)
                        video_transforms.append(gaussian_blur)

                    if apply_color_jitter:
                        color_jitter = transforms.ColorJitter(
                            brightness=0.2,
                            contrast=0.2,
                            saturation=0.2,
                            hue=0.05
                        )
                        video_transforms.append(color_jitter)

                    # Fixed parameters for deterministic visualization
                    fog_intensity = 0.15 if apply_fog else 0
                    noise_factor = 0.02 if apply_noise else 0
                    rain_drop_length = 3 if apply_rain else 0
                    rain_drop_count = 30 if apply_rain else 0

                # For normalization like the dataset uses
                to_tensor = transforms.ToTensor()
                resize = transforms.Resize((112, 112))
                normalizer = transforms.Normalize(
                    mean=[0.43216, 0.394666, 0.37645],
                    std=[0.22803, 0.22145, 0.216989]
                )

                # Process each frame
                transformed_frames = []
                for frame_idx in smart_indices:
                    video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    success, frame = video.read()
                    if not success:
                        continue

                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Convert to tensor and resize
                    frame_tensor = to_tensor(frame.astype(np.float32) / 255.0)
                    frame_tensor = resize(frame_tensor)

                    # Apply normalization
                    frame_tensor = normalizer(frame_tensor)

                    # Apply augmentations
                    if apply_augmentation:
                        # Apply torchvision transforms
                        for transform in video_transforms:
                            frame_tensor = transform(frame_tensor)

                        # Apply custom effects - need to denormalize first
                        denorm = transforms.Normalize(
                            mean=[-0.43216/0.22803, -0.394666/0.22145, -0.37645/0.216989],
                            std=[1/0.22803, 1/0.22145, 1/0.216989]
                        )

                        # Apply fog effect if enabled
                        if apply_fog:
                            tmp = denorm(frame_tensor)
                            tmp = add_fog(tmp, fog_intensity=fog_intensity)
                            frame_tensor = normalizer(tmp)

                        # Apply noise effect if enabled
                        if apply_noise:
                            tmp = denorm(frame_tensor)
                            tmp = add_noise(tmp, noise_factor=noise_factor)
                            frame_tensor = normalizer(tmp)

                        # Apply rain effect if enabled and function is available
                        if apply_rain and 'simulate_rain' in globals():
                            tmp = denorm(frame_tensor)
                            tmp = simulate_rain(
                                tmp,
                                drop_length=rain_drop_length,
                                drop_count=rain_drop_count,
                                seed=frame_idx
                            )
                            frame_tensor = normalizer(tmp)

                    # Apply horizontal flip if enabled
                    if apply_horizontal_flip:
                        frame_tensor = torch.flip(frame_tensor, dims=[2])

                    transformed_frames.append(frame_tensor)

                video.release()

                # Check if we got any frames
                if not transformed_frames:
                    raise ValueError("No frames were successfully loaded")

                # Convert frames to tensor of shape (C, T, H, W)
                frames_tensor = torch.stack(transformed_frames, dim=1)

                # Get video FPS for time calculations
                cap = cv2.VideoCapture(video_file)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()

                # Display each frame
                for i, (frame_tensor, frame_idx) in enumerate(zip(transformed_frames, smart_indices)):
                    if i >= args.frames_per_video:
                        break

                    # Convert tensor to image for display
                    denorm = transforms.Normalize(
                        mean=[-0.43216/0.22803, -0.394666/0.22145, -0.37645/0.216989],
                        std=[1/0.22803, 1/0.22145, 1/0.216989]
                    )
                    display_tensor = denorm(frame_tensor)
                    display_tensor = torch.clamp(display_tensor, 0, 1)
                    frame_image = display_tensor.permute(1, 2, 0).numpy()

                    # Get frame time and check if it's near accident
                    is_accident_frame = False
                    frame_title = f"Frame {frame_idx}"

                    if fps > 0:
                        frame_time = frame_idx / fps
                        frame_title = f"Frame {frame_idx} ({frame_time:.2f}s)"

                        if time_of_event is not None:
                            # Check if this frame is within 0.5 seconds of the accident
                            if abs(frame_time - time_of_event) < 0.5:
                                is_accident_frame = True
                                frame_title += " ⚠️"  # Add warning symbol

                    axes[config_idx, i].imshow(frame_image)
                    axes[config_idx, i].set_title(frame_title,
                                                color='red' if is_accident_frame else 'black')
                    axes[config_idx, i].axis('off')

                    # Add red border to accident frames
                    if is_accident_frame:
                        for spine in axes[config_idx, i].spines.values():
                            spine.set_color('red')
                            spine.set_linewidth(2)

                return frames_tensor

            # Use our custom visualization function
            try:
                visualize_with_custom_frames()
            except Exception as e:
                print(f"  Error visualizing frames: {e}")
                # Fill in empty cells with error message
                for frame_idx in range(args.frames_per_video):
                    axes[config_idx, frame_idx].text(0.5, 0.5, f"Error: {str(e)[:40]}...",
                                                 horizontalalignment='center',
                                                 verticalalignment='center')
                    axes[config_idx, frame_idx].axis('off')

        # Add row labels for augmentation types
        for idx, config in enumerate(augmentation_configs):
            axes[idx, 0].set_ylabel(config["name"], fontsize=10, rotation=90, labelpad=15)

        # Save the figure
        plt.tight_layout()
        fig.suptitle(f"Video ID: {video_id} - Augmentation Comparison", fontsize=16, y=1.02)
        plt.savefig(os.path.join(args.output_dir, f"video_{video_id}_augmentations.png"),
                   dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Clean up temporary file
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)

    print(f"Visualization complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
