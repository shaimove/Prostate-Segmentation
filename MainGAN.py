import torch
import torch.cuda
from torch import nn
from torch.utils import data
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt
from datasets import DatasetProstate
from models import UNet,Discriminator,AE
import utils 
import losses
import Training

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% Create dataset
# Load raw dataset and split to train/validation/test, need to be done only once!
#(mean,std) = utils.ReadSplitData('../PROMISE12/',split = [0.7,0.15,0.15])
stats = [0.41836,0.245641]
batch_size_train = 32
batch_size_validation = 32

# define dataset and dataloader for training
train_dataset = DatasetProstate('../PROMISE12/',stats,mode='train')
train_loader = data.DataLoader(train_dataset,batch_size=batch_size_train,shuffle=True)

# define dataset and dataloader for validation
validation_dataset = DatasetProstate('../PROMISE12/',stats,mode='validation')
validation_loader = data.DataLoader(validation_dataset,batch_size=batch_size_validation,shuffle=True)


#%% Training
# Define training parameters
n_epochs = 200
lambda_recon = 200
input_dim = 1
real_dim = 1
target_shape = 256
lr = 0.0001
display_step = 10
path = '../PROMISE12/pix2pix results/'

# pack parameters to send to training
params = [n_epochs,lambda_recon,input_dim,target_shape,real_dim,lr,display_step,path]

# Define loss function for generator and discriminator
adv_criterion = nn.BCEWithLogitsLoss() 
recon_criterion = losses.DiceLoss()

# Define models
gen = UNet(input_dim, real_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)

disc = Discriminator(input_dim + real_dim).to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

# pack models to send to training
models_opt_loss = [adv_criterion,recon_criterion,gen,gen_opt,disc,disc_opt]
datasets = [train_dataset,train_loader,validation_dataset,validation_loader]

# Print models structure
print('Generator model')
summary(gen, (1, 256, 256))

print('Discriminator model')
summary(disc, [(1, 256, 256),(1, 256, 256)])

#%% Phase 1: Training the pix2pix U-net
models_opt_loss = Training.TrainerPix2Pix(params, models_opt_loss,datasets)
adv_criterion,recon_criterion,gen,gen_opt,disc,disc_opt = models_opt_loss


#%% Phase 2: train only the Discriminator on curropted images
# Define AE 










#%% Phase 3: freeze the discriminator weight, and fine-tuning the U-net
# Load trained network
path_model = path + "pix2pix.pth"

gen.load_state_dict(torch.load(path_model['gen']))
gen_opt.load_state_dict(torch.load(path_model['gen_opt']))
disc.load_state_dict(torch.load(path_model['disc']))
disc_opt.load_state_dict(torch.load(path_model['disc_opt']))






































