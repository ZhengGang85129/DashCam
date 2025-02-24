import torch
import torch.nn as nn
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

from dataset import VideoDataset
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

class ObjectDetector(nn.Module):
    vehicle_label = [2, 3, 4, 6, 8] # class labels for 'car', 'bicycle', 'bus', 'motocycle', 'truck' in FasterRCNN_ResNet50_FPN_Weights  
    def __init__(self, score_threshold: float = 0.8)->None:
        super(ObjectDetector, self).__init__()
        self.weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights = self.weights, progress = False)
        self.model.eval()
        self.class_name = self.weights.meta["categories"]
        self.transforms = self.weights.transforms()
        self.score_threshold = score_threshold
         
    def crop_by_box(self, img, box: List[int]):
        """
        """
        x_min, y_min, x_max, y_max = [int(coord) for coord in box]
        
        return img[:, y_min:y_max, x_min:x_max]
    
    def plot_object(self, objects: List[torch.Tensor])->None:
        
        for i in range(len(objects)):   
            plt.imshow(objects[i].permute(1, 2, 0).detach().cpu().numpy())
            plt.savefig(f'object-{i:02d}.png') 
    def process_video(self) -> None:
        """
        #Test function#
        """
        video_path = './dataset/train/extracted/00227.mp4'
        frames = load_video(video_path)
        img = self.transforms(frames[0].permute(2, 0, 1))
        detection_outputs = self.model(img.unsqueeze(0))
        output = detection_outputs[0] 
        score_threshold = .8
        #vehicle = torch.tensor(['car', 'bicycle', 'person', 'bus', 'motorcycle', 'truck'])
        class_mask = torch.isin(output['labels'], 
                    torch.tensor(self.vehicle_label, dtype=torch.int64))
        
        score_mask = output['scores'] > score_threshold
        class_mask = class_mask[score_mask] 
        boxes = output['boxes'][score_mask] 
        objects = [self.crop_by_box(img, box) for box in boxes ]
        #self.plot_object(objects)  -> can use this function to plot the object "cropped" by the faster r-cnn
        
        return objects

    def forward(self, x: torch.Tensor) -> None:
        """
        Provide the objects list in each frame using faster R-CNN
        Args: 
            x: torch.Tensor: Tensor of shape (batch_size, n_frames, channels, height, width)
        """
        
        n_frames = x.shape[1]
        objects = [] 
        for frame_index in range(n_frames):
            img = self.transforms(x[:, frame_index, :, :, :].permute(0, 2, 3, 1))    
            #print(img.shape)
            img = img.permute(0, 3, 1, 2)
            detection_outputs = self.model(img) 
            output = detection_outputs[0]
            class_mask = torch.isin(output['labels'], 
                    torch.tensor(self.vehicle_label, dtype=torch.int64))
            score_mask = output['scores'] > self.score_threshold
            class_mask = class_mask[score_mask] 
        
            boxes = output['boxes'][score_mask][class_mask]
            print(boxes) 
            print([self.crop_by_box(img, box) for box in boxes ])
            #objects.append([self.crop_by_box(img, box) for box in boxes ])
        ##self.plot_object(objects)  -> can use this function to plot the object "cropped" by the faster r-cnn
        #return torch.stack(objects, dim = 0).permute()
        #print()
        #return objects

     




if __name__ == '__main__':
    #dogs_with_boxes = [
    #    draw_bounding_boxes(img, boxes = boxes[class_mask], width=4)
    #]
    #show(dogs_with_boxes)
    start = time.time()
    detector = ObjectDetector()
    dataset = VideoDataset()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4
    )
    
    for batch in dataloader:
        with torch.no_grad():
            #print(batch.shape)
            detector(batch)
        break
    
    #print(detector.process_video())
    print(time.time() - start, ' sec')