from skimage import io, transform
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from os import listdir
from os.path import isfile, join
import os
from PIL import Image
import re

# class DrivingDataset(Dataset):
    
#     def __init__(self, root_dir, categorical = False, classes=-1, transform=None):
#         """
#         root_dir (string): Directory with all the images.
#         transform (callable, optional): Optional transform to be applied on a sample.
#         """
#         self.root_dir = root_dir
#         self.transform = transform
#         self.filenames = [f for f in listdir(self.root_dir) if f.endswith('jpg')]
#         self.categorical = categorical
#         self.classes = classes
        
#     def __len__(self):
#         return len(self.filenames)
        
#     def __getitem__(self, idx):
#         basename = self.filenames[idx]
#         img_name = os.path.join(self.root_dir, basename)
#         image = io.imread(img_name)

#         m = re.search('expert_([0-9]+)_([0-9]+)_([-+]?\d*\.\d+|\d+).jpg', basename)
#         steering_command = np.array(float(m.group(3)), dtype=np.float32)

#         if self.categorical:
#             steering_command = int(((steering_command + 1.0)/2.0) * (self.classes - 1)) 
            
#         if self.transform:
#             image = self.transform(image)
        
#         return {'image': image, 'cmd': steering_command}




class DrivingDataset(Dataset):
    def __init__(self, root_dir, categorical=False, classes=20, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.categorical = categorical
        self.classes = classes
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)
        
        # Updated regex pattern to handle scientific notation
        m = re.match(r'expert_(\d+)_(\d+)_(-?\d+\.?\d*(e-?\d+)?).jpg', self.image_files[idx])
        
        if m:
            steering_command = np.array(float(m.group(3)), dtype=np.float32)
        else:
            raise ValueError(f"Filename '{self.image_files[idx]}' does not match the expected pattern 'expert_<iteration>_<timestep>_<steering>.jpg'")

        if self.transform:
            image = self.transform(image)

        if self.categorical:
            # Convert continuous steering values to class labels
            steering_command = np.clip(steering_command, -1, 1)  # Ensure steering_command is within [-1, 1]
            class_label = int((steering_command + 1) / 2 * (self.classes - 1))  # Map [-1, 1] to [0, classes-1]
            return {'image': image, 'cmd': class_label}
        
        return {'image': image, 'cmd': steering_command}
