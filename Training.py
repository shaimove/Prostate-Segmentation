# Training.py
import torch
import torch.cuda
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import utils
import losses

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% TrainerPix2Pix

def TrainerPix2Pix(params, models_opt_loss,datasets):
    # Unpack parameters for training and models
    n_epochs,lambda_recon,input_dim,target_shape,real_dim,lr,display_step,path = params
    adv_criterion,recon_criterion,gen,gen_opt,disc,disc_opt = models_opt_loss
    train_dataset,train_loader,validation_dataset,validation_loader = datasets

    # Loss vectors
    train_disc_loss_vec = []; train_gen_loss_vec = []
    valid_disc_loss_vec = []; valid_gen_loss_vec = []
    
    
    for epoch in range(n_epochs):        
        ##################
        ### TRAIN LOOP ###
        ##################
        gen.train(); disc.train()
        train_disc_loss = 0; train_gen_loss = 0
        
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
            fake,gen_loss = losses.get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon)
            gen_loss.backward() # Update gradients
            gen_opt.step() # Update optimizer
            
            ### Loss for batch ###        
            train_disc_loss += disc_loss.item() / len(train_loader)
            train_gen_loss += gen_loss.item() / len(train_loader)
            
        # append train losses to vectors
        train_disc_loss_vec.append(train_disc_loss)
        train_gen_loss_vec.append(train_gen_loss)
        
        #######################
        ### VALIDATION LOOP ###
        #######################
        gen.eval(); disc.eval()
        valid_disc_loss = 0; valid_gen_loss = 0
           
        with torch.no_grad():
            for batch in validation_loader:
                # get data from batch and send to GPU
                condition = batch['image'].to(device)
                real = batch['Segmentation'].to(device)
        
                ### get losses ###
                disc_loss = losses.get_disc_loss(gen, disc, real, condition, adv_criterion)
                fake,gen_loss = losses.get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon)
                
                ### Loss for batch ### 
                valid_disc_loss += disc_loss.item() / len(validation_loader)
                valid_gen_loss += gen_loss.item() / len(validation_loader)
                
                
        # append validation losses to vectors   
        valid_disc_loss_vec.append(valid_disc_loss)
        valid_gen_loss_vec.append(valid_gen_loss)
        
        ######################################
        ### Epoch Summary and save results ###
        ######################################
        print('Epoch %d: Generator loss: Train: %.3f, Validation: %.3f, Discriminator loss: Train: %.3f, Validation: %.3f'
              % (epoch, train_gen_loss,valid_gen_loss, train_disc_loss,valid_disc_loss))
        utils.show_images(condition, real, fake, 3, epoch, path,
                          size = (input_dim, target_shape, target_shape))
        
        
    ### save the model at the end ###
    path_model = path + "pix2pix.pth"
    torch.save({'gen': gen.state_dict(),
                'gen_opt': gen_opt.state_dict(),
                'disc': disc.state_dict(),
                'disc_opt': disc_opt.state_dict()
                }, path_model)       
        
    ### Plot training results ###
    plt.figure()
    plt.plot(range(n_epochs),train_gen_loss_vec,label='Training Generator Loss')
    plt.plot(range(n_epochs),valid_gen_loss_vec,label='Validation Generator loss')
    plt.grid(); plt.xlabel('Number of epochs'); plt.ylabel('Loss')
    plt.title('Generator Loss for pix2pix network'); plt.legend()
    result_path = path + 'Generator results.png'
    plt.savefig(result_path)
    plt.close('all')
    
    plt.figure()
    plt.plot(range(n_epochs),train_disc_loss_vec,label='Training Discriminator Loss')
    plt.plot(range(n_epochs),valid_disc_loss_vec,label='Validation Discriminator loss')
    plt.grid(); plt.xlabel('Number of epochs'); plt.ylabel('Loss'); plt.legend()
    plt.title('Discriminator Loss for pix2pix network')
    result_path = path + 'Discriminator results.png'
    plt.savefig(result_path)
    
    # pack back 
    models_opt_loss = [adv_criterion,recon_criterion,gen,gen_opt,disc,disc_opt]
    
    return models_opt_loss


#%% TrainerUnet

def TrainerUnet(params, models_opt_loss,datasets):
    # Unpack parameters for training and models
    n_epochs,input_dim,target_shape,real_dim,lr,path = params
    unet,unet_opt,criterion = models_opt_loss
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
            output = unet(condition)
            unet_loss = criterion(output,real)
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
                output = unet(condition)
                unet_loss = criterion(output,real)
                valid_loss += unet_loss.item() / len(validation_loader)
        
        valid_loss_vec.append(valid_loss)
        
        ######################################
        ### Epoch Summary and save results ###
        ######################################
        print('Epoch %d: Train Dice loss: %.3f, Validation Dice loss %.3f' 
              % (epoch, train_loss, valid_loss))
        utils.show_images(condition, real, output, 3, epoch, path,
                          size = (input_dim, target_shape, target_shape))
        
        
    # save the model at the end
    path_model = path + "unet.pth"
    torch.save({'unet': unet.state_dict(),
                'unet_opt': unet_opt.state_dict(),
                }, path_model)       
        
    # Plot training results
    plt.figure()
    plt.plot(range(n_epochs),train_loss_vec,label='Training Dice Loss')
    plt.plot(range(n_epochs),valid_loss_vec,label='Validation Dice Loss')
    plt.grid(); plt.xlabel('Number of epochs'); plt.ylabel('Loss')
    plt.title('Loss for U-net network'); plt.legend()
    
    result_path = path + 'results.png'
    plt.savefig(result_path)
    
    # pack back
    models_opt_loss = [unet,unet_opt,criterion]
    
    return models_opt_loss