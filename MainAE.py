import torch
import torch.cuda
from torch import nn
from torch.utils import data
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt
from datasets import DatasetProstate
from models import UNet,AE
import utils 
import losses
import Training
import TrainingDiscAE
import Training_Finetune

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# create folders to save results
path = '../PROMISE12/'
folders = ['unet','AE','unetSR']
paths = utils.CreateFolders(path,folders)

#%% Create dataset
# Load raw dataset and split to train/validation/test, need to be done only once!
#(mean,std) = utils.ReadSplitData('../PROMISE12/',split = [0.7,0.15,0.15])
stats = [0.41836,0.245641]
batch_size_train = 32
batch_size_validation = 32

# Training
# Define training parameters
n_epochs = 200
input_dim = 1
target_shape = 256
real_dim = 1
lr = 0.0001

# Define U-net 
unet = UNet(input_dim, real_dim).to(device)
unet_opt = torch.optim.Adam(unet.parameters(), lr=lr)

print('U-net model')
summary(unet, (1, 256, 256))

# define AE
ae = AE(input_dim, real_dim).to(device)
ae_opt = torch.optim.Adam(ae.parameters(), lr=lr)

print('AE model')
summary(ae, (1, 256, 256))

#%% Phase 1: train U-net 
# define dataset and dataloader for training
train_dataset = DatasetProstate('../PROMISE12/',stats,mode='train')
train_loader = data.DataLoader(train_dataset,batch_size=batch_size_train,shuffle=True)

# define dataset and dataloader for validation
validation_dataset = DatasetProstate('../PROMISE12/',stats,mode='validation')
validation_loader = data.DataLoader(validation_dataset,batch_size=batch_size_validation,shuffle=True)

# pack parameters to send to training
params = [n_epochs,input_dim,target_shape,real_dim,lr,paths[0]]

# Define loss function for unet
criterion = losses.DiceLoss()

# pack models to send to training
models_opt_loss = [unet,unet_opt,criterion]
datasets = [train_dataset,train_loader,validation_dataset,validation_loader]

# Train the model
models_opt_loss = Training.TrainerUnet(params, models_opt_loss,datasets)


#%% Phase 2: Train VA on curropted data
# define dataset and dataloader for training
train_dataset = DatasetProstate('../PROMISE12/',stats,mode='train',TrainAE = True)
train_loader = data.DataLoader(train_dataset,batch_size=batch_size_train,shuffle=True)

# define dataset and dataloader for validation
validation_dataset = DatasetProstate('../PROMISE12/',stats,mode='validation',TrainAE = True)
validation_loader = data.DataLoader(validation_dataset,batch_size=batch_size_validation,shuffle=True)

# define loss function
criterion = losses.DiceLoss()

# pack params, models and datasets
params = [n_epochs,input_dim,target_shape,real_dim,lr,paths[1]]
models_opt_loss = [ae,ae_opt,criterion]
datasets = [train_dataset,train_loader,validation_dataset,validation_loader]

# Train the model
models_opt_loss = TrainingDiscAE.TrainAE(params, models_opt_loss,datasets)

#%% Phase 3: Fine tunning 
path_uent = paths[0] + "/unet.pth"
path_ae = paths[1] + "/ae.pth"
unet.load_state_dict(torch.load(path_uent)['unet'])
unet_opt.load_state_dict(torch.load(path_uent)['unet_opt'])
ae.load_state_dict(torch.load(path_ae)['ae'])
ae_opt.load_state_dict(torch.load(path_ae)['ae_opt'])

# define dataset and dataloader for training
train_dataset = DatasetProstate('../PROMISE12/',stats,mode='train')
train_loader = data.DataLoader(train_dataset,batch_size=batch_size_train,shuffle=True)

# define dataset and dataloader for validation
validation_dataset = DatasetProstate('../PROMISE12/',stats,mode='validation')
validation_loader = data.DataLoader(validation_dataset,batch_size=batch_size_validation,shuffle=True)

# define loss function
criterion = nn.MSELoss()
lambda_reco = 1
lambda_latent = 1
batch_size_train = 16
batch_size_validation = 16

# pack params, models and datasets
params = [n_epochs,input_dim,target_shape,real_dim,lr,lambda_reco,lambda_latent,paths[2]]
models_opt_loss = [unet,unet_opt,ae,ae_opt,criterion]
datasets = [train_dataset,train_loader,validation_dataset,validation_loader]

# Train the model
models_opt_loss = Training_Finetune.TrainerUnetFT(params, models_opt_loss,datasets)
