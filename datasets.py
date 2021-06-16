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
    
    def __init__(self,path,stats,mode='train',TrainAE = False, Discriminator = False):
        # root folder is training or testing
        self.path = path 
        
        # mode of dataset (train,validation,test)
        self.mode = mode
        self.TrainAE = TrainAE
        self.Discriminator = Discriminator
        
        # read file according to mode
        if self.TrainAE:
            self.Y_corr = np.load(self.path + 'Y_corr_' + self.mode + '.npy')
            self.Y = np.load(self.path + 'Y_' + self.mode + '.npy')
            
        elif self.Discriminator:
            self.X = np.load(self.path + 'X_' + self.mode + '.npy')
            self.Y = np.load(self.path + 'Y_' + self.mode + '.npy')
            self.Y_corr = np.load(self.path + 'Y_corr_' + self.mode + '.npy')
            
        else:
            self.X = np.load(self.path + 'X_' + self.mode + '.npy')
            self.Y = np.load(self.path + 'Y_' + self.mode + '.npy')
        

            
        # stats: mean and std of the training set
        self.mean = stats[0]
        self.std = stats[1]
        
        # initiate transforms
        self.TransformsX()
        self.TransformsY()
        
        
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
        return self.Y.shape[0]
    
    def __getitem__(self,idx):
        # False for MRI images, and True for curropted segmentation images
        if self.TrainAE: 
            seg_corr,seg_map = self.Y_corr[idx,::], self.Y[idx,::]
            seg_corr = self.transformsY(seg_corr).float()
            seg_map = self.transformsY(seg_map).float()
            sample = {'Segmentation_curr': seg_corr, 'Segmentation': seg_map}
            
        elif self.Discriminator:
            image,seg_corr,seg_map = self.X[idx,::],self.Y_corr[idx,::],self.Y[idx,::]
            image = self.transformsX(image).float()
            seg_corr = self.transformsY(seg_corr).float()
            seg_map = self.transformsY(seg_map).float()
            sample = {'image': image,'Segmentation_curr': seg_corr,
                      'Segmentation': seg_map}
            
        else:
            image,seg_map = self.X[idx,::], self.Y[idx,::]
            image = self.transformsX(image).float()
            seg_map = self.transformsY(seg_map).float()
            sample = {'image': image,'Segmentation': seg_map}
        
        
        return sample

