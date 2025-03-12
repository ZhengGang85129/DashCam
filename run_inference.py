import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import argparse
import os
from tqdm import tqdm
from model import CollisionPredictionModel  # Assume your trained model is in 'model.py'

# Define input transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Adjust to match training input size
    transforms.ToTensor(),
])

def preprocess_video(video_path, num_frames=30):
    """
    Extract frames from the video, preprocess them, and return as a tensor.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(num_frames):
        frame_id = int(total_frames * (i / num_frames))  # Sample frames uniformly
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame)
        frames.append(frame)
        frame_count += 1

    cap.release()

    if len(frames) < num_frames:
        # Pad with black frames if needed
        while len(frames) < num_frames:
            frames.append(torch.zeros((3, 224, 224)))

    return torch.stack(frames).unsqueeze(0)  # Shape: (1, num_frames, 3, 224, 224)

def load_model(model_path, device):
    """
    Load the trained model for inference.
    """
    model = CollisionPredictionModel()  # Replace with your model architecture
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def infer(model, video_path, device):
    """
    Run inference on a video and return the probability of a collision.
    """
    input_tensor = preprocess_video(video_path).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probability = torch.sigmoid(output).item()  # Assuming binary classification
    return probability

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collision Prediction Inference")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model checkpoint")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(args.model, device)
    probability = infer(model, args.video, device)

    print(f"Collision Probability: {probability:.4f}")
