# Test.py
import torch
import torch.cuda
from torch import nn
from torch.utils import data
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt
from datasets import DatasetProstate
from models import UNet
import utils 
import losses


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% Load test dataset
# define states
stats = [0.41836,0.245641]
batch_size_test = 32
input_dim = 1; real_dim = 1;

# define dataset and dataloader for training
test_dataset = DatasetProstate('../PROMISE12/',stats,mode='test')
test_loader = data.DataLoader(test_dataset,batch_size=batch_size_test,shuffle=True)

# upload generator from GAN model
path_GAN = '../PROMISE12/GAN finetune results/unet.pth'
gen = UNet(input_dim, real_dim).to(device)
gen.load_state_dict(torch.load(path_GAN)['gen'])

# upload unet from original paper
path_AE = '../PROMISE12/AE finetune results/'
unet = UNet(input_dim, real_dim).to(device)
unet.load_state_dict(torch.load(path_AE)['unet'])

#%% Loop over results for GAN model
loss_GAN = 0
loss_AE = 0

for batch in test_loader:
    with torch.no_grad():
        # get data from batch and send to GPU
        condition = batch['image'].to(device)
        real = batch['Segmentation'].to(device)
        
        # output from gen and unet
        output_gen = gen(condition)
        output_unet = unet(condition)
        
        # Dice loss
        loss_gen = losses.DiceLoss(output_gen,real)
        loss_unet = losses.DiceLoss(output_unet,real)
        
        # add to loss
        loss_GAN +=  loss_gen.item() / len(test_loader)
        loss_AE +=  loss_unet.item() / len(test_loader)

                
# print results
print('Dice Loss for AE model is %.4f' % loss_AE)
print('Dice Loss for GAN model is %.4f' % loss_GAN)














