# TrainingDiscAE.py
import torch
import torch.cuda
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import utils
import losses

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% Trainer for the AE

def TrainAE(params, models_opt_loss,datasets):
    # Unpack parameters for training and models
    n_epochs,input_dim,target_shape,real_dim,lr,path = params
    ae,ae_opt,criterion = models_opt_loss
    train_dataset,train_loader,validation_dataset,validation_loader = datasets

    # Loss vectors
    train_loss_vec = []; valid_loss_vec = []

    for epoch in range(n_epochs):
        ##################
        ### TRAIN LOOP ###
        ##################
        ae.train(); train_loss = 0
        
        for batch in train_loader:
            # get data from batch and send to GPU
            seg_curr = batch['Segmentation_curr'].to(device)
            real = batch['Segmentation'].to(device)
            
            ### Update model ###
            ae_opt.zero_grad() 
            Ex,output = ae(seg_curr)
            ae_loss = criterion(output,real)
            ae_loss.backward(retain_graph=True) 
            ae_opt.step() 

            ### Loss ###        
            train_loss += ae_loss.item() / len(train_loader)

        train_loss_vec.append(train_loss)
        
        #######################
        ### VALIDATION LOOP ###
        #######################
        ae.eval(); valid_loss = 0
        with torch.no_grad():
            for batch in validation_loader:
                # get data from batch and send to GPU
                seg_curr = batch['Segmentation_curr'].to(device)
                real = batch['Segmentation'].to(device)
                
                ### Loss ### 
                Ex,output = ae(seg_curr)
                ae_loss = criterion(output,real)
                valid_loss += ae_loss.item() / len(validation_loader)
        
        valid_loss_vec.append(valid_loss)
        
        ######################################
        ### Epoch Summary and save results ###
        ######################################
        print('Epoch %d: Train Dice loss: %.3f, Validation Dice loss %.3f' 
              % (epoch, train_loss, valid_loss))
        utils.show_images(seg_curr, real, output, 3, epoch, path,
                          size = (input_dim, target_shape, target_shape))
        
        
    # save the model at the end
    path_model = path + "/ae.pth"
    torch.save({'ae': ae.state_dict(),
                'ae_opt': ae_opt.state_dict(),
                }, path_model)       
        
    # Plot training results
    plt.figure()
    plt.plot(range(n_epochs),train_loss_vec,label='Training Dice Loss')
    plt.plot(range(n_epochs),valid_loss_vec,label='Validation Dice Loss')
    plt.grid(); plt.xlabel('Number of epochs'); plt.ylabel('Loss')
    plt.title('Loss for AE network'); plt.legend()
    
    result_path = path + 'results.png'
    plt.savefig(result_path)
    
    # pack back
    models_opt_loss = [ae,ae_opt,criterion]
    
    return models_opt_loss



#%% Trainer of the discriminator 
def TrainDiscriminator(params, models_opt_loss,datasets):
    # Unpack parameters for training and models
    n_epochs,input_dim,target_shape,real_dim,lr,path = params
    disc,disc_opt,criterion = models_opt_loss
    train_dataset,train_loader,validation_dataset,validation_loader = datasets
    
    # Loss vectors
    train_loss_vec = []; valid_loss_vec = []

    for epoch in range(n_epochs):
        ##################
        ### TRAIN LOOP ###
        ##################
        disc.train(); train_loss = 0
        
        for batch in train_loader:
            # get data from batch and send to GPU
            condition = batch['image'].to(device)
            seg_curr = batch['Segmentation_curr'].to(device)
            real = batch['Segmentation'].to(device)
            
            ### Update model ###
            disc_opt.zero_grad() 
            # loss for curropted segementation map
            disc_fake_hat = disc(seg_curr,condition)
            disc_fake_loss = criterion(disc_fake_hat,torch.zeros_like(disc_fake_hat))
            
            # loss for read segementation map
            disc_real_hat = disc(real,condition)
            disc_real_loss = criterion(disc_real_hat,torch.ones_like(disc_fake_hat))
            
            # final loss
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            # step
            disc_loss.backward(retain_graph=True) 
            disc_opt.step() 

            ### Loss ###        
            train_loss += disc_loss.item() / len(train_loader)

        train_loss_vec.append(train_loss)
        
        #######################
        ### VALIDATION LOOP ###
        #######################
        disc.eval(); valid_loss = 0
        with torch.no_grad():
            for batch in validation_loader:
                # get data from batch and send to GPU
                condition = batch['image'].to(device)
                seg_curr = batch['Segmentation_curr'].to(device)
                real = batch['Segmentation'].to(device)
                
                ### Loss ### 
                # loss for curropted segementation map
                disc_fake_hat = disc(seg_curr,condition)
                disc_fake_loss = criterion(disc_fake_hat,torch.zeros_like(disc_fake_hat))
            
                # loss for read segementation map
                disc_real_hat = disc(real,condition)
                disc_real_loss = criterion(disc_real_hat,torch.ones_like(disc_fake_hat))
            
                # final loss and step
                disc_loss = (disc_fake_loss + disc_real_loss) / 2
                valid_loss += disc_loss.item() / len(validation_loader)
        
        valid_loss_vec.append(valid_loss)
        
        ######################################
        ### Epoch Summary and save results ###
        ######################################
        print('Epoch %d: Train Dice loss: %.3f, Validation Dice loss %.3f' 
              % (epoch, train_loss, valid_loss))
        utils.show_images(condition, real, seg_curr, 3, epoch, path,
                          size = (input_dim, target_shape, target_shape))
        
        
    # save the model at the end
    path_model = path + "/disc.pth"
    torch.save({'disc': disc.state_dict(),
                'disc_opt': disc_opt.state_dict(),
                }, path_model)       
        
    # Plot training results
    plt.figure()
    plt.plot(range(n_epochs),train_loss_vec,label='Training Dice Loss')
    plt.plot(range(n_epochs),valid_loss_vec,label='Validation Dice Loss')
    plt.grid(); plt.xlabel('Number of epochs'); plt.ylabel('Loss')
    plt.title('Loss for Discriminator'); plt.legend()
    
    result_path = path + 'results.png'
    plt.savefig(result_path)
    
    # pack back
    models_opt_loss = [disc,disc_opt,criterion]
    
    return models_opt_loss





