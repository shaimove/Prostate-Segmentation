# Training_Finetune.py
import torch
import torch.cuda
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import utils
import losses

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#%% Fine tunning Trainer for GAN
def TrainerGAN_FT(params, models_opt_loss,datasets):
    # Unpack parameters for training and models
    n_epochs,input_dim,target_shape,real_dim,lr,lambda_reco,path = params
    gen,gen_opt,disc,disc_opt,adv_criterion,recon_criterion = models_opt_loss
    train_dataset,train_loader,validation_dataset,validation_loader = datasets
    
    # Loss vectors
    train_loss_vec = []; valid_loss_vec = []
    
    for epoch in range(n_epochs):
        ##################
        ### TRAIN LOOP ###
        ##################
        gen.train(); train_loss = 0
        
        for batch in train_loader:
            # get data from batch and send to GPU
            condition = batch['image'].to(device)
            real = batch['Segmentation'].to(device)

            ### Update model ###
            gen_opt.zero_grad() 
            fake,gen_loss = losses.GAN_FT_Loss(gen,disc,real,condition,adv_criterion,recon_criterion,lambda_reco)
            gen_loss.backward(retain_graph=True) 
            gen_opt.step()
            
            ### Loss for batch ###        
            train_loss += gen_loss.item() / len(train_loader)
        
        train_loss_vec.append(train_loss)
        
        #######################
        ### VALIDATION LOOP ###
        #######################
        gen.eval(); valid_loss = 0
        with torch.no_grad():
            for batch in validation_loader:
                # get data from batch and send to GPU
                condition = batch['image'].to(device)
                real = batch['Segmentation'].to(device)
                
                ### Loss ### 
                fake,gen_loss = losses.GAN_FT_Loss(gen,disc,real,condition,adv_criterion,recon_criterion,lambda_reco)
                valid_loss += gen_loss.item() / len(validation_loader)
        
        valid_loss_vec.append(valid_loss)
        
        ######################################
        ### Epoch Summary and save results ###
        ######################################
        print('Epoch %d: Train loss: %.3f, Validation loss %.3f' 
              % (epoch, train_loss, valid_loss))
        utils.show_images(condition, real, fake, 3, epoch, path,
                          size = (input_dim, target_shape, target_shape))

    # save the model at the end
    path_model = path + "/pix2pixSR.pth"
    torch.save({'gen': gen.state_dict(),
                'gen_opt': gen_opt.state_dict(),
                }, path_model)       
        
    # Plot training results
    plt.figure()
    plt.plot(range(n_epochs),train_loss_vec,label='Training Loss')
    plt.plot(range(n_epochs),valid_loss_vec,label='Validation Loss')
    plt.grid(); plt.xlabel('Number of epochs'); plt.ylabel('Loss')
    plt.title('Loss for GAN SR Network'); plt.legend()
    
    result_path = path + 'results.png'
    plt.savefig(result_path)




#%% Fine tunning Trainer for AE

def TrainerUnetFT(params, models_opt_loss,datasets):
    # Unpack parameters for training and models
    n_epochs,input_dim,target_shape,real_dim,lr,lambda_reco,lambda_latent,path = params
    unet,unet_opt,ae,ae_opt,criterion = models_opt_loss
    train_dataset,train_loader,validation_dataset,validation_loader = datasets
    
    # Loss vectors
    train_loss_vec = []; valid_loss_vec = []
    
    for epoch in range(n_epochs):
        ##################
        ### TRAIN LOOP ###
        ##################
        unet.train(); train_loss = 0
        
        for batch in train_loader:
            # get data from batch and send to GPU
            condition = batch['image'].to(device)
            real = batch['Segmentation'].to(device)
            
            ### Update model ###
            unet_opt.zero_grad() 
            output,unet_loss = losses.Unet_FT_Loss(unet,ae,real,condition,criterion,lambda_reco,lambda_latent)
            unet_loss.backward(retain_graph=True) 
            unet_opt.step() 

            ### Loss ###        
            train_loss += unet_loss.item() / len(train_loader)
    
        train_loss_vec.append(train_loss)
        
        #######################
        ### VALIDATION LOOP ###
        #######################
        unet.eval(); valid_loss = 0
        with torch.no_grad():
            for batch in validation_loader:
                # get data from batch and send to GPU
                condition = batch['image'].to(device)
                real = batch['Segmentation'].to(device)
                
                ### Loss ### 
                fake,unet_loss = losses.Unet_FT_Loss(unet,ae,real,condition,criterion,lambda_reco,lambda_latent)
                valid_loss += unet_loss.item() / len(validation_loader)
        
        valid_loss_vec.append(valid_loss)
        
        ######################################
        ### Epoch Summary and save results ###
        ######################################
        print('Epoch %d: Train loss: %.3f, Validation loss %.3f' 
              % (epoch, train_loss, valid_loss))
        utils.show_images(condition, real, output, 3, epoch, path,
                          size = (input_dim, target_shape, target_shape))
    
    
    # save the model at the end
    path_model = path + "/unetSR.pth"
    torch.save({'unet': unet.state_dict(),
                'unet_opt': unet_opt.state_dict(),
                }, path_model)       
        
    # Plot training results
    plt.figure()
    plt.plot(range(n_epochs),train_loss_vec,label='Training Loss')
    plt.plot(range(n_epochs),valid_loss_vec,label='Validation Loss')
    plt.grid(); plt.xlabel('Number of epochs'); plt.ylabel('Loss')
    plt.title('Loss for U-net SR Network'); plt.legend()
    
    result_path = path + 'results.png'
    plt.savefig(result_path)


    
    return None