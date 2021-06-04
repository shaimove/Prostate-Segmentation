import os
import numpy as np
import torch
import torch.cuda
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms


# beacuse the data set is so small, we will read all the data at init stage
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DatasetMRI(Dataset):
    
    def __init__(self,path,stats,mode='train'):
        # root folder is training or testing
        self.path = path 
        
        # mode of dataset (train,validation,test)
        self.mode = mode
        
        # read file according to mode
        self.X = np.load(self.path + 'X_' + self.mode + '.npy')
        self.Y = np.load(self.path + 'Y_' + self.mode + '.npy')
        
        # initiate transform
        self.createTransforms()
        
        # stats: mean and std of the training set
        self.mean = stats[0]
        self.std = stats[1]
        
    def createTransforms(self):
        # create basic transforms
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[self.mean],std=[self.std])])
    
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self,idx):
        # get image path and label
        image,seg_map = self.X[idx,::], self.Y[idx,::]
        
        # preform transforms
        if self.transforms:
            image = self.transforms(image).unsqueeze(0).float()
            seg_map = self.transforms(seg_map).unsqueeze(0).float()
        
        # create dictionary
        sample = {'image': image, 'Segmentation': seg_map}
        
        
        return sample

