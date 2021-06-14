# Test.py
import torch
import torch.cuda
from torch.utils import data

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

# upload generator from GAN model without Shape regularization
path_GAN = '../PROMISE12/pix2pix/pix2pix.pth'
gen = UNet(input_dim, real_dim).to(device)
gen.load_state_dict(torch.load(path_GAN)['gen'])

# upload generator from GAN model with Shape regularization
path_GANSR = '../PROMISE12/pix2pixSR/pix2pixSR.pth'
genSR = UNet(input_dim, real_dim).to(device)
genSR.load_state_dict(torch.load(path_GANSR)['gen'])

# upload unet from original paper without Shape regularization
path_unet = '../PROMISE12/unet/unet.pth'
unet = UNet(input_dim, real_dim).to(device)
unet.load_state_dict(torch.load(path_unet)['unet'])

# upload unet from original paper with Shape regularization
path_unetSR = '../PROMISE12/unetSR/unetSR.pth'
unetSR = UNet(input_dim, real_dim).to(device)
unetSR.load_state_dict(torch.load(path_unetSR)['unet'])

#%% Loop over results for GAN model
loss_gen = 0; loss_genSR = 0
loss_unet = 0; loss_unetSR = 0

for batch in test_loader:
    with torch.no_grad():
        # get data from batch and send to GPU
        condition = batch['image'].to(device)
        real = batch['Segmentation'].to(device)
        
        # output from gen and unet
        output_gen = gen(condition)
        output_unet = unet(condition)
        output_genSR = genSR(condition)
        output_unetSR = unetSR(condition)
        
        # Dice loss
        loss_gen_batch = losses.DiceLoss(output_gen,real)
        loss_unet_batch = losses.DiceLoss(output_unet,real)
        loss_genSR_batch = losses.DiceLoss(output_genSR,real)
        loss_unetSR_batch = losses.DiceLoss(output_unetSR,real)
        
        # add to loss
        loss_gen +=  loss_gen_batch.item() / len(test_loader)
        loss_unet +=  loss_unet_batch.item() / len(test_loader)
        loss_genSR += loss_genSR_batch.item() / len(test_loader)
        loss_unetSR += loss_unetSR_batch.item() / len(test_loader)
                
# print results
print('Dice Loss for GAN without SR is %.4f' % loss_gen)
print('Dice Loss for unet without SR is %.4f' % loss_unet)
print('Dice Loss for GAN with SR is %.4f' % loss_genSR)
print('Dice Loss for unet with SR is %.4f' % loss_unetSR)














