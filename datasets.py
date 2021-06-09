import os
import numpy as np
import torch
import torch.cuda
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms


# beacuse the data set is so small, we will read all the data at init stage
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DatasetProstate(Dataset):
    
    def __init__(self,path,stats,mode='train'):
        # root folder is training or testing
        self.path = path 
        
        # mode of dataset (train,validation,test)
        self.mode = mode
        
        # read file according to mode
        self.X = np.load(self.path + 'X_' + self.mode + '.npy')
        self.Y = np.load(self.path + 'Y_' + self.mode + '.npy')
        
        # stats: mean and std of the training set
        self.mean = stats[0]
        self.std = stats[1]
        
        # initiate transforms
        self.TransformsX()
        self.TransformsY()
        
        # Do we want to perform tranforms? 
        self.transforms = True
        
        
        
    def TransformsX(self):
        # create basic transforms
        self.transformsX = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[self.mean],std=[self.std])])
    
    def TransformsY(self):
        # create basic transforms
        self.transformsY = transforms.Compose([
            transforms.ToTensor()])
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self,idx):
        # get image path and label
        image,seg_map = self.X[idx,::], self.Y[idx,::]
        
        # preform transforms
        if self.transforms:
            image = self.transformsX(image).float()
            seg_map = self.transformsY(seg_map).float()
        
        # create dictionary
        sample = {'image': image, 'Segmentation': seg_map}
        
        
        return sample

