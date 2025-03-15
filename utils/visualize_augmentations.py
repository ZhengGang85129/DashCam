import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import os
import random
from pathlib import Path
import argparse
from PIL import Image

# Import your augmented dataset class - adjust import path as needed
# from utils.augmented_dataset import AugmentedVideoDataset

# Define functions for creating various augmentations
def add_noise(img, noise_factor=0.1):
    """Add random noise to an image tensor"""
    noise = torch.randn_like(img) * noise_factor
    noisy_img = img + noise
    return torch.clamp(noisy_img, 0., 1.)

def add_fog(img, fog_intensity=0.3):
    """Simulate fog effect by adding a bright overlay with reduced contrast"""
    fog = torch.ones_like(img) * fog_intensity
    foggy_img = img * (1 - fog_intensity) + fog
    return torch.clamp(foggy_img, 0., 1.)

def simulate_rain(img, drop_length=20, drop_width=1, drop_count=20):
    """Simulate rain by adding random streaks"""
    c, h, w = img.shape
    rain_img = img.clone()
    
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
    
    return torch.clamp(rain_img, 0., 1.)

def visualize_augmentations():
    parser = argparse.ArgumentParser(description='Visualize data augmentations on dashcam videos')
    parser.add_argument('--video_path', type=str, required=True, help='Path to a sample video file')
    parser.add_argument('--output_dir', type=str, default='augmentation_samples', help='Directory to save visualization results')
    parser.add_argument('--frames', type=int, default=5, help='Number of frames to sample from the video')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Open the video
    video = cv2.VideoCapture(args.video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        print(f"Error: Could not read frames from {args.video_path}")
        return
        
    # Calculate interval to sample frames
    interval = max(1, total_frames // args.frames)
    
    # Base transforms
    base_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),  # Larger size for better visualization
    ])
    
    # Normalization transform - separate to visualize before/after
    normalize = transforms.Normalize(
        mean=[0.43216, 0.394666, 0.37645], 
        std=[0.22803, 0.22145, 0.216989]
    )
    
    # Basic augmentation transforms
    color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
    gaussian_blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
    h_flip = transforms.RandomHorizontalFlip(p=1.0)  # Always flip for visualization
    
    # Frame extraction and augmentation
    frames = []
    frame_indices = list(range(0, total_frames, interval))[:args.frames]
    
    for frame_idx in frame_indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = video.read()
        if not success:
            continue
            
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = base_transforms(frame.astype(np.float32) / 255.0)
        frames.append(frame_tensor)
    
    video.release()
    
    if not frames:
        print("No frames were successfully read from the video.")
        return
    
    # Create visualizations for each frame
    for i, frame in enumerate(frames):
        # Create a figure with subplots
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        plt.subplots_adjust(hspace=0.3)
        
        # Original frame
        axes[0, 0].imshow(frame.permute(1, 2, 0).numpy())
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        # Normalized frame (what the model sees as input)
        normalized = normalize(frame)
        # Denormalize for visualization
        denorm = transforms.Normalize(
            mean=[-0.43216/0.22803, -0.394666/0.22145, -0.37645/0.216989],
            std=[1/0.22803, 1/0.22145, 1/0.216989]
        )
        axes[0, 1].imshow(denorm(normalized).permute(1, 2, 0).numpy().clip(0, 1))
        axes[0, 1].set_title('Normalized Input')
        axes[0, 1].axis('off')
        
        # Color Jitter
        axes[0, 2].imshow(color_jitter(frame).permute(1, 2, 0).numpy())
        axes[0, 2].set_title('Color Jitter')
        axes[0, 2].axis('off')
        
        # Gaussian Blur
        axes[1, 0].imshow(gaussian_blur(frame).permute(1, 2, 0).numpy())
        axes[1, 0].set_title('Gaussian Blur')
        axes[1, 0].axis('off')
        
        # Horizontal Flip
        axes[1, 1].imshow(h_flip(frame).permute(1, 2, 0).numpy())
        axes[1, 1].set_title('Horizontal Flip')
        axes[1, 1].axis('off')
        
        # Noise
        axes[1, 2].imshow(add_noise(frame, noise_factor=0.1).permute(1, 2, 0).numpy())
        axes[1, 2].set_title('Noise Added')
        axes[1, 2].axis('off')
        
        # Fog
        axes[2, 0].imshow(add_fog(frame, fog_intensity=0.3).permute(1, 2, 0).numpy())
        axes[2, 0].set_title('Fog Effect')
        axes[2, 0].axis('off')
        
        # Rain
        axes[2, 1].imshow(simulate_rain(frame, drop_length=20, drop_width=1, drop_count=30).permute(1, 2, 0).numpy())
        axes[2, 1].set_title('Rain Effect')
        axes[2, 1].axis('off')
        
        # Combined augmentations
        combined = frame.clone()
        # Apply random augmentations
        if random.random() > 0.5:
            combined = color_jitter(combined)
        if random.random() > 0.5:
            combined = gaussian_blur(combined)
        if random.random() > 0.5:
            combined = h_flip(combined)
        if random.random() > 0.3:
            combined = add_noise(combined, noise_factor=0.08)
        if random.random() > 0.3:
            combined = add_fog(combined, fog_intensity=0.2)
        if random.random() > 0.2:
            combined = simulate_rain(combined, drop_count=random.randint(20, 40))
            
        axes[2, 2].imshow(combined.permute(1, 2, 0).numpy())
        axes[2, 2].set_title('Combined Effects')
        axes[2, 2].axis('off')
        
        # Save figure
        fig.suptitle(f'Frame {frame_idx} Augmentations', fontsize=16)
        plt.savefig(os.path.join(args.output_dir, f'frame_{i}_augmentations.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Visualization completed. Images saved to {args.output_dir}")

if __name__ == "__main__":
    visualize_augmentations()
