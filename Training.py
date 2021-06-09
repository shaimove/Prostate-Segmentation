# Training.py
import torch
import torch.cuda
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import utils
import losses

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def TrainerPix2Pix(params, models_opt_loss,datasets):
    # Unpack parameters for training and models
    n_epochs,lambda_recon,input_dim,target_shape,real_dim,lr,display_step,path = params
    adv_criterion,recon_criterion,gen,gen_opt,disc,disc_opt = models_opt_loss
    train_dataset,train_loader = datasets

    
    generator_loss = []
    discriminator_loss = []
    
    
    for epoch in range(n_epochs):
        # declare loss variabels for every epoch
        mean_generator_loss = 0
        mean_discriminator_loss = 0
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
            fake,gen_loss = losses.get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon)
            gen_loss.backward() # Update gradients
            gen_opt.step() # Update optimizer
            
            ### Loss ###        
            # Keep track of the average discriminator loss
            mean_discriminator_loss += disc_loss.item() / len(train_loader)
            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item() / len(train_loader)
            
        ### Visualization every epoch ###
        print('Epoch %d: Generator loss: %.2f, Discriminator loss: %.3f'
              % (epoch, mean_generator_loss, mean_discriminator_loss))
        utils.show_images(condition, real, fake, 3, epoch, path,
                          size = (input_dim, target_shape, target_shape))
        
        generator_loss.append(mean_generator_loss)
        discriminator_loss.append(mean_discriminator_loss)
    
    # save the model at the end
    path_model = path + "pix2pix.pth"
    torch.save({'gen': gen.state_dict(),
                'gen_opt': gen_opt.state_dict(),
                'disc': disc.state_dict(),
                'disc_opt': disc_opt.state_dict()
                }, path_model)       
        
    # Plot training results
    plt.figure()
    plt.plot(range(n_epochs),generator_loss,label='Generator Loss')
    plt.plot(range(n_epochs),discriminator_loss,label='Discriminator Loss')
    plt.grid(); plt.xlabel('Number of epochs'); plt.ylabel('Loss')
    plt.title('Loss for pix2pix network')
    plt.legend()
    result_path = path + 'results.png'
    plt.savefig(result_path)
    
    models_opt_loss = [adv_criterion,recon_criterion,gen,gen_opt,disc,disc_opt]
    
    return models_opt_loss


#%%
def TrainerUnet(params, models_opt_loss,datasets):
    # Unpack parameters for training and models
    n_epochs,input_dim,target_shape,real_dim,lr,display_step,path = params
    unet,unet_opt,criterion = models_opt_loss
    train_dataset,train_loader = datasets

    
    loss_vec = []
    
    
    for epoch in range(n_epochs):
        # declare loss variabels for every epoch
        mean_loss = 0
        ##################
        ### TRAIN LOOP ###
        ##################
        for batch in train_loader:
            # get data from batch and send to GPU
            condition = batch['image'].to(device)
            real = batch['Segmentation'].to(device)
            
            ### Update discriminator ###
            unet_opt.zero_grad() # zeros gradient before calculating loss
            output = unet(condition)
            unet_loss = criterion(output,real)
            unet_loss.backward(retain_graph=True) # Update gradients
            unet_opt.step() # Update optimizer

            ### Loss ###        
            # Keep track of the average dice loss
            mean_loss += unet_loss.item() / len(train_loader)

            
        ### Visualization every epoch ###
        print('Epoch %d: Mean Dice loss: %.3f' % (epoch, mean_loss))
        utils.show_images(condition, real, output, 3, epoch, path,
                          size = (input_dim, target_shape, target_shape))
        
        loss_vec.append(mean_loss)
    
    # save the model at the end
    path_model = path + "unet.pth"
    torch.save({'unet': unet.state_dict(),
                'unet_opt': unet_opt.state_dict(),
                }, path_model)       
        
    # Plot training results
    plt.figure()
    plt.plot(range(n_epochs),loss_vec,label='Dice Loss')
    plt.grid(); plt.xlabel('Number of epochs'); plt.ylabel('Loss')
    plt.title('Loss for U-net network')
    plt.legend()
    result_path = path + 'results.png'
    plt.savefig(result_path)
    
    models_opt_loss = [unet,unet_opt,criterion]
    
    return models_opt_loss