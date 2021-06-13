#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
from models import UNet
import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#%% Define parameters
path = '../PROMISE12/'
path_model = '../PROMISE12/unet_early_epoch/unet.pth'
stats = [0.41836,0.245641]
input_dim = 1; real_dim = 1; 

# load images
x_train = np.load(path + 'X_train.npy')
y_train = np.load(path + 'Y_train.npy')
# load model
unet = UNet(input_dim, real_dim,AE=True).to(device)
unet.load_state_dict(torch.load(path_model)['unet'])
# prepare to run
num_of_img = y_train.shape[0]
y_corr = np.zeros(y_train.shape)

#%% stage 2: read the images, normalize and go throw the network
for i in tqdm.tqdm(range(num_of_img)):
    x = x_train[i,::]
    y = y_train[i,:,:,0]
    
    if np.sum(y) == 0:
        y = np.expand_dims(np.expand_dims(y,0),3)
        y_corr[i,::] = y
        
    else:
        # normalize and convet to tensor
        x = torch.Tensor((x - stats[0]) / stats[0])
        # prepare size to U-net and U-net
        x = torch.unsqueeze(x.permute(2,0,1),0).to(device)
        y_out = unet(x)
        # Prepare back to save as numpy array
        y_out = y_out.cpu().squeeze(0).detach().numpy()
        y_out = np.expand_dims(y_out,3)
        y_out = np.where(y_out > 0.3,y_out,0)
        # add to y_corr
        y_corr[i,::] = y_out

# save images
np.save(path + 'Y_corr.npy',y_corr)

#%% Plot images
num_of_img_dis = 5

ind = np.random.randint(0,num_of_img,num_of_img_dis)
fig = plt.figure(figsize = (12,12))

for p,i in enumerate(ind):
    y_img = y_train[i,:,:,0]
    y_corr_img = y_corr[i,:,:,0]
    y_display = np.concatenate((y_img, y_corr_img), axis=1)
    
    fig.add_subplot(num_of_img_dis,1,p+1); plt.imshow(y_display,cmap='gray')

fig.suptitle('Ground Truth Segmentation / Early epoch Segmentation')
    
    
