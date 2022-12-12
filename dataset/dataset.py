import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import pydicom
import os
import numpy as np
import pandas as pd
from skimage.io import imread
class INBreastDataset(Dataset):
    def __init__(self, root: str, img_dir: str):
        self.img_dir = img_dir
        self.img_data = pd.read_csv(img_dir)
    
    def __len__(self):
        return len(self.img_data['Lesion Annotation Status'])
    
    def __getitem__(self, idx: int):
        img_path = os.path.join(self.root, self.img_data.iloc[idx, 0])
        img = pydicom.dcmread(img_path)
        img = np.array(img.pixel_array, dtype = np.float32)
        img = torch.from_numpy(img)
        label = self.img_data['Lesion Annotation Status'][idx]
        return img, label





# class MIASDataset(Dataset):
#     def __init__(self, root: str, img_dir: str):
#         self.img_dir = img_dir
#         self.img_data = pd.read_csv(img_dir)
    
#     def __len__(self):
#         return len(self.img_data['Lesion Annotation Status'])
    
#     def __getitem__(self, idx: int):
#         img_path = os.path.join(self.root, self.img_data.iloc[idx, 0])
#         img = pydicom.dcmread(img_path)
#         img = np.array(img.pixel_array, dtype = np.float32)
#         img = torch.from_numpy(img)
#         label = self.img_data['Lesion Annotation Status'][idx]
#         return img, label


