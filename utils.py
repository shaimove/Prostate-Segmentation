import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid



def ReadSplitData(path_read,path_write=None,split = [0.7,0.15,0.15]):
    '''
    The following function read the raw dataset which is stored in 4D npy file,
    and then split it to 3 parts: Training, Validation and test sets. 
    

    Parameters
    ----------
    path_read : string
        The path to find the MRI scans and segemntation maps, in 4D npy file.
    path_write : string
        The path to write the MRI scans and segemntation maps, in 4D npy file.
    split : list size 3
        porpotions of training, validation and test set, sum to 1.

    Returns
    -------
    stats : list, (mean,std)
        mean and std of the whole dataset, not only training. 

    '''
    # if path_write isn't given, path_write = path_read
    if path_write is None:
        path_write = path_read

    # Read data
    X = np.load(path_read + 'X.npy')
    Y = np.load(path_read + 'Y.npy')
    
    # create random indxes array to split the data
    num = X.shape[0]
    vec_ind = np.arange(num)
    np.random.shuffle(np.arange(num))
    
    # take indexes to every group
    ind_train = vec_ind[:int(split[0]*num)]
    ind_validation = vec_ind[int(split[0]*num):int(split[0]*num) + int(split[1]*num)]
    ind_test = vec_ind[int(split[0]*num) + int(split[1]*num):]
    
    # split the data and save in different files 
    np.save(path_write + 'X_train.npy', X[ind_train,::])
    np.save(path_write + 'Y_train.npy', Y[ind_train,::])
        
    np.save(path_write + 'X_validation.npy',X[ind_validation,::])
    np.save(path_write + 'Y_validation.npy',Y[ind_validation,::])
    
    np.save(path_write + 'X_test.npy', X[ind_test,::])
    np.save(path_write + 'Y_test.npy', Y[ind_test,::])

    # before throwing the dataset to the dark side, calculate mean and std
    mean = np.mean(X[ind_train,::])
    std = np.std(X[ind_train,::])
    stats = [mean,std]
    
    # now delete X and Y
    X = []; Y = []
    

    return stats


def show_tensor_images(image_tensor, num_images=9, size=(1, 256, 256)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=3)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze(),cmap='gray')
    plt.show()

