import torch
import torch.cuda
from torch import nn
import torch.nn.functional as F
from torch.utils import data

import numpy as np
import matplotlib.pyplot as plt
from datasets import DatasetMRI
from models import UNet,Discriminator
import utils 
import losses

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% Create dataset
# Load raw dataset and split to train/validation/test, need to be done only once!
#(mean,std) = utils.ReadSplitData('../PROMISE12/',split = [0.7,0.15,0.15])
stats = [0.41836,0.245641]
batch_size_train = 4
batch_size_validation = 4

# define dataset and dataloader for training
train_dataset = DatasetMRI('../PROMISE12/',stats,mode='train')
train_loader = data.DataLoader(train_dataset,batch_size=batch_size_train,shuffle=True)

# define dataset and dataloader for validation
validation_dataset = DatasetMRI('../PROMISE12/',stats,mode='validation')
validation_loader = data.DataLoader(validation_dataset,batch_size=batch_size_validation,shuffle=True)


#%% Training
# Define training parameters
n_epochs = 20
lambda_recon = 200
input_dim = 1
real_dim = 1
lr = 0.001

# Define loss function for generator and discriminator
adv_criterion = nn.BCEWithLogitsLoss() 
recon_criterion = losses.DiceLoss()

# Define models 
gen = UNet(input_dim, real_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)

disc = Discriminator(input_dim + real_dim).to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

#%%

mean_generator_loss = 0
mean_discriminator_loss = 0


for epoch in range(n_epochs):
    ##################
    ### TRAIN LOOP ###
    ##################
    for batch in train_loader:
        # get data from batch and send to GPU
        condition = batch['image'].to(device)
        real = batch['Segmentation'].to(device)
        
        ### Update discriminator ###
        disc_opt.zero_grad() # zeros gradient before calculating loss
        disc_loss = losses.get_disc_loss(gen, disc, real, condition, adv_criterion) # loss
        disc_loss.backward(retain_graph=True) # Update gradients
        disc_opt.step() # Update optimizer
        
        ### Update generator ###
        gen_opt.zero_grad() # zeros gradient before calculating loss
        gen_loss = losses.get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon)
        gen_loss.backward() # Update gradients
        gen_opt.step() # Update optimizer
        
        
#%% Plot training results
plt.figure()
plt.plot(range(n_epochs),mean_generator_loss,label='Generator Loss')
plt.plot(range(n_epochs),mean_discriminator_loss,label='Discriminator Loss')
plt.grid(); plt.xlabel('Number of epochs'); plt.ylabel('Loss')
plt.title('Loss for pix2pix network')
plt.legend()





































