import torch
import torch.cuda
from torch import nn
from torch.utils import data
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt
from datasets import DatasetProstate
from models import UNet,Discriminator
import utils 
import losses
import Training
import TrainingDiscAE
import Training_Finetune

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# create folders to save results
path = '../PROMISE12/'
folders = ['pix2pix','Disc','pix2pixSR']
paths = utils.CreateFolders(path,folders)

#%% Create dataset
# Load raw dataset and split to train/validation/test, need to be done only once!
#(mean,std) = utils.ReadSplitData('../PROMISE12/',split = [0.7,0.15,0.15])
stats = [0.41836,0.245641]
batch_size_train = 32
batch_size_validation = 32

# Define training parameters
n_epochs = 200
lambda_recon = 200
input_dim = 1
real_dim = 1
target_shape = 256
lr = 0.0001

# Define models
gen = UNet(input_dim, real_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)

disc = Discriminator(input_dim + real_dim).to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

# Print models structure
print('Generator model')
summary(gen, (1, 256, 256))

print('Discriminator model')
summary(disc, [(1, 256, 256),(1, 256, 256)])

#%% Phase 1: Training the pix2pix U-net
# define dataset and dataloader for training
train_dataset = DatasetProstate('../PROMISE12/',stats,mode='train')
train_loader = data.DataLoader(train_dataset,batch_size=batch_size_train,shuffle=True)

# define dataset and dataloader for validation
validation_dataset = DatasetProstate('../PROMISE12/',stats,mode='validation')
validation_loader = data.DataLoader(validation_dataset,batch_size=batch_size_validation,shuffle=True)

# pack parameters to send to training
params = [n_epochs,lambda_recon,input_dim,target_shape,real_dim,lr,paths[0]]

# Define loss function for generator and discriminator
adv_criterion = nn.BCEWithLogitsLoss() 
recon_criterion = losses.DiceLoss()

# pack models to send to training
models_opt_loss = [adv_criterion,recon_criterion,gen,gen_opt,disc,disc_opt]
datasets = [train_dataset,train_loader,validation_dataset,validation_loader]

# training
models_opt_loss = Training.TrainerPix2Pix(params, models_opt_loss,datasets)
adv_criterion,recon_criterion,gen,gen_opt,disc,disc_opt = models_opt_loss


#%% Phase 2: train only the Discriminator on curropted images
# load disriminator
path_model = paths[0] + "/pix2pix.pth"
disc.load_state_dict(torch.load(path_model)['disc'])
disc_opt.load_state_dict(torch.load(path_model)['disc_opt'])

# define dataset and dataloader for training
train_dataset = DatasetProstate('../PROMISE12/',stats,mode='train', Discriminator = True)
train_loader = data.DataLoader(train_dataset,batch_size=batch_size_train,shuffle=True)

# define dataset and dataloader for validation
validation_dataset = DatasetProstate('../PROMISE12/',stats,mode='validation', Discriminator = True)
validation_loader = data.DataLoader(validation_dataset,batch_size=batch_size_validation,shuffle=True)

# Define loss function 
criterion = losses.DiceLoss()

# pack models to send to training
params = [n_epochs,input_dim,target_shape,real_dim,lr,paths[1]]
models_opt_loss = [disc,disc_opt,criterion]
datasets = [train_dataset,train_loader,validation_dataset,validation_loader]

# Train the model
models_opt_loss = TrainingDiscAE.TrainDiscriminator(params, models_opt_loss,datasets)


#%% Phase 3: freeze the discriminator weight, and fine-tuning the U-net
# Load trained network
path_gen = paths[0] + "/pix2pix.pth"
path_disc = paths[1] + "/disc.pth"
gen.load_state_dict(torch.load(path_gen)['gen'])
gen_opt.load_state_dict(torch.load(path_gen)['gen_opt'])
disc.load_state_dict(torch.load(path_disc)['disc'])
disc_opt.load_state_dict(torch.load(path_disc)['disc_opt'])

# Define loss function 
recon_criterion = losses.DiceLoss()
adv_criterion = nn.BCEWithLogitsLoss() 
lambda_reco = 5
batch_size_train = 16
batch_size_validation = 16

# define dataset and dataloader for training
train_dataset = DatasetProstate('../PROMISE12/',stats,mode='train')
train_loader = data.DataLoader(train_dataset,batch_size=batch_size_train,shuffle=True)

# define dataset and dataloader for validation
validation_dataset = DatasetProstate('../PROMISE12/',stats,mode='validation')
validation_loader = data.DataLoader(validation_dataset,batch_size=batch_size_validation,shuffle=True)


# pack models to send to training
params = [n_epochs,input_dim,target_shape,real_dim,lr,lambda_reco,paths[2]]
models_opt_loss = [gen,gen_opt,disc,disc_opt,adv_criterion,recon_criterion]
datasets = [train_dataset,train_loader,validation_dataset,validation_loader]

# Train the model
models_opt_loss = Training_Finetune.TrainerGAN_FT(params, models_opt_loss,datasets)
































