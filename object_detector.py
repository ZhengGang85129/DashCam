import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.utils import draw_bounding_boxes
import os
from typing import List
import cv2
import time
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import numpy as np
os.environ['TORCH_HOME'] = os.getcwd() #will download model weights to your current work directory

"""
https://pytorch.org/vision/main/auto_examples/others/plot_visualization_utils.html#sphx-glr-auto-examples-others-plot-visualization-utils-py
"""


def load_video(video_path: str, resize_shape: List[int] = [224, 224])-> List[torch.Tensor]:
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize frame
        #frame = cv2.resize(frame, resize_shape)
        # Apply transforms
        frames.append(torch.from_numpy(frame))
            
        frame_count += 1
        
    cap.release()
    
    # Ensure we have enough frames
    return frames 

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.savefig('123.png')

class FeatureExtractor:
    def __init__(self, )->None:
        
        self.weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights = self.weights, progress = False)
        self.model.eval()
        self.class_name = self.weights.meta["categories"]
        self.transforms = self.weights.transforms()
    
    def process_video(self) -> None:
        video_path = './dataset/train/extracted/00227.mp4'
        frames = load_video(video_path)
        img = self.transforms(frames[0].permute(2, 0, 1))
        detection_outputs = self.model(img.unsqueeze(0))
        output = detection_outputs[0] 
        score_threshold = .8
        #vehicle = torch.tensor(['car', 'bicycle', 'person', 'bus', 'motorcycle', 'truck'])
        vehicle_label = [2, 3, 4, 6, 8] # class labels for 'car', 'bicycle', 'bus', 'motocycle', 'truck' in FasterRCNN_ResNet50_FPN_Weights  
        class_mask = torch.isin(output['labels'], 
                    torch.tensor(vehicle_label, dtype=torch.int64))
        
        score_mask = output['scores'] > score_threshold
        class_mask = class_mask[score_mask] 
        boxes = output['boxes'][score_mask] 
        
        pass





if __name__ == '__main__':
    #dogs_with_boxes = [
    #    draw_bounding_boxes(img, boxes = boxes[class_mask], width=4)
    #]
    #show(dogs_with_boxes)
    
