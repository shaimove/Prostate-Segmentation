# Training.py
import torch
import torch.cuda
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import utils


def TrainerPix2Pix(params, models_opt_loss,datasets):
    # Unpack parameters for training and models
    n_epochs,lambda_recon,input_dim,target_shape,real_dim,lr,display_step = params
    adv_criterion,recon_criterion,gen,gen_opt,disc,disc_opt = models_opt_loss
    train_dataset,train_loader = datasets

    
    # Define iterators
    mean_generator_loss = 0
    mean_discriminator_loss = 0
    cur_step = 0
    
        
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
            fake,gen_loss = losses.get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon)
            gen_loss.backward() # Update gradients
            gen_opt.step() # Update optimizer
            
            # Keep track of the average discriminator loss
            mean_discriminator_loss += disc_loss.item() / display_step
            # Keep track of the average generator loss
            mean_generator_loss += gen_loss.item() / display_step
            
            ### Visualization code ###
            if cur_step % display_step == 0:
                if cur_step > 0:
                    print(f"Epoch {epoch}: Step {cur_step}: Generator (U-Net) loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}")
                else:
                    print("Pretrained initial state")
                    utils.show_tensor_images(condition, size=(input_dim, target_shape, target_shape))
                    utils.show_tensor_images(real, size=(real_dim, target_shape, target_shape))
                    utils.show_tensor_images(fake, size=(real_dim, target_shape, target_shape))
                    mean_generator_loss = 0
                    mean_discriminator_loss = 0                
            cur_step += 1
    
    # save the model at the end
    torch.save({'gen': gen.state_dict(),
                'gen_opt': gen_opt.state_dict(),
                'disc': disc.state_dict(),
                'disc_opt': disc_opt.state_dict()
                }, "pix2pix.pth")       
        
    #%% Plot training results
    plt.figure()
    plt.plot(range(n_epochs),mean_generator_loss,label='Generator Loss')
    plt.plot(range(n_epochs),mean_discriminator_loss,label='Discriminator Loss')
    plt.grid(); plt.xlabel('Number of epochs'); plt.ylabel('Loss')
    plt.title('Loss for pix2pix network')
    plt.legend()



