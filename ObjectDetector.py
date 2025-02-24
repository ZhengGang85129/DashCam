import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from FeatureExtractor import FeatureExtractor as extractor
import os
from typing import List, Tuple
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import numpy as np
os.environ['TORCH_HOME'] = os.getcwd() #will download model weights to your current work directory

"""
https://pytorch.org/vision/main/auto_examples/others/plot_visualization_utils.html#sphx-glr-auto-examples-others-plot-visualization-utils-py
"""


class ObjectDetector(nn.Module):
    vehicle_label = [2, 3, 4, 6, 8] # class labels for 'car', 'bicycle', 'bus', 'motocycle', 'truck' in FasterRCNN_ResNet50_FPN_Weights  
    def __init__(self, score_threshold: float = 0.8, max_num_objects:int = 20)->None:
        """
        Args:
            score_threshold(float)
            max_num_objects(int): Maximum number of objects with highest score provided by this object detector.
        """
        super(ObjectDetector, self).__init__()
        self.weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights = self.weights, progress = False)
        self.model.eval()
        self.class_name = self.weights.meta["categories"]
        self.transforms = self.weights.transforms()
        self.score_threshold = score_threshold
        self.max_num_objects = max_num_objects
        self.extractor = extractor()
        self.extractor.eval()
         
    def crop_by_box(self, img, box: List[int]):
        """
        """
        x_min, y_min, x_max, y_max = [int(coord) for coord in box]
        
        return img[:, y_min:y_max, x_min:x_max]
    
    def plot_object(self, objects: List[torch.Tensor])->None:
        
        for i in range(len(objects)):   
            plt.imshow(objects[i].permute(1, 2, 0).detach().cpu().numpy())
            plt.savefig(f'object-{i:02d}.png') 

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Provide the objects list in each frame using faster R-CNN
        Args: 
            x: torch.Tensor: Tensor of shape (batch_size, channels, height, width)
        Return:
            Objects(torch.Tensor), batch_size, num_objects, 4: each box associated with the object given by the faster r-cnn. dimension 4 represent x_min, y_min, x_max, y_max, respectively.
            Object_mask(torch.Tensor): 0 indicate that there is an object, and 1 indicate that attention mechanism will ignore this position.
        """
        batch_size = x.shape[0]
        
        images = self.transforms(x.permute(0, 2, 3, 1))    
        #print(img.shape)
        images = images.permute(0, 3, 1, 2)
        detection_outputs = self.model(images)
        
        
        objects = []
        objects_mask = []
        objects_features = []
        for batch_index in range(batch_size):
            object_holder = torch.zeros((self.max_num_objects, 4))
            
            output = detection_outputs[batch_index]

            class_mask = torch.isin(output['labels'], torch.tensor(self.vehicle_label, dtype=torch.int64))
            
            comb_mask = class_mask & (output['scores'] > self.score_threshold) 
            
            comb_mask = class_mask & (output['scores'] > self.score_threshold)
            #raise ValueError() 
            k = min(comb_mask.sum(), self.max_num_objects) 
            
            scores = output['scores'][comb_mask]
            boxes = output['boxes']

            _, top_k_indices = torch.topk(scores, k = k) 
            
            n_objects = len(top_k_indices)
            
            boxes = boxes[top_k_indices]
            object_holder[:n_objects] = boxes
            
            obj_mask_holder = torch.ones((self.max_num_objects)) # Will mask the empty position which does not have object. 
            obj_mask_holder[n_objects:] = 0
            objects.append(object_holder)
            objects_mask.append(obj_mask_holder)
            
            feature_holder = torch.zeros((self.max_num_objects, 4096))
            
            for obj_idx in range(n_objects):
                print(images[batch_index, :, int(object_holder[obj_idx][1].item()):int(object_holder[obj_idx][3].item()), int(object_holder[obj_idx][0].item()):int(object_holder[obj_idx][2].item())].shape)
                feature = self.extractor(images[batch_index, :, int(object_holder[obj_idx][1].item()):int(object_holder[obj_idx][3].item()), int(object_holder[obj_idx][0].item()):int(object_holder[obj_idx][2].item())])
                feature_holder[obj_idx] = feature 
            objects_features.append(feature_holder) 
        Objects = torch.stack(objects, dim = 0)
        Object_features = torch.stack(objects_features, dim = 0)
        Objects_mask = torch.stack(objects_mask, dim = 0) 
        
        return Objects, Object_features, Objects_mask 

     




if __name__ == '__main__':
    from dataset import VideoDataset
    import time
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
        n_frames = batch.shape[1]
        for frame_idx in range(n_frames): 
            with torch.no_grad():
                objects, objects_features, objects_mask = detector(batch[:, frame_idx,:,:,:])
            #for video in batch:
            #print(batch.shape)
            #    detector(video[0])
        break
    
    #print(detector.process_video())
    print(time.time() - start, ' sec')