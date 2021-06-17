import torch
from torch import nn

#%% DiceLoss

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

#%% Generator_Loss

def Generator_Loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon):
    '''
    Return the loss of the generator given inputs.
    Parameters:
        gen: the generator; takes the condition and returns potential images
        disc: the discriminator; takes images and the condition and
          returns real/fake prediction matrices
        real: the real images (e.g. maps) to be used to evaluate the reconstruction
        condition: the source images (e.g. satellite imagery) which are used to produce the real images
        adv_criterion: the adversarial loss function; takes the discriminator 
                  predictions and the true labels and returns a adversarial loss
        recon_criterion: the reconstruction loss function; takes the generator 
                    outputs and the real images and returns a reconstructuion loss
        lambda_recon: the degree to which the reconstruction loss should be weighted in the sum
    '''

    # Generate the fake images, based on the conditions.
    fake = gen(condition)
    
    # Evaluate the fake images and the condition with the discriminator.
    disc_fake = disc(fake,condition)
    
    # Calculate the adversarial loss.
    gen_adv_loss = adv_criterion(disc_fake,torch.ones_like(disc_fake))
    
    # Calculate the reconstruction loss.
    gen_rec_loss = recon_criterion(real,fake)
    
    # Add the two losses, weighting the reconstruction loss appropriately.
    gen_loss = gen_adv_loss + lambda_recon * gen_rec_loss

    return fake,gen_loss


#%% Discriminator_Loss

def Discriminator_Loss(gen, disc, real, condition, adv_criterion):
    '''
    Return the loss of the discriminator given inputs.
    Parameters:
        gen: the generator; takes the condition and returns potential images
        disc: the discriminator; takes images and the condition and
          returns real/fake prediction matrices
        real: the real images (e.g. maps) to be used to evaluate the reconstruction
        condition: the source images (e.g. satellite imagery) which are used to produce the real images
        adv_criterion: the adversarial loss function; takes the discriminator 
                  predictions and the true labels and returns a adversarial loss
    '''
    # Generate the fake images, based on the conditions.
    with torch.no_grad():
        fake = gen(condition)
        
    # Evaluate the fake images and the condition with the discriminator.
    disc_fake_hat = disc(fake.detach(), condition) # Detach generator
    
    # Calculate the adversarial loss
    disc_fake_loss = adv_criterion(disc_fake_hat, torch.zeros_like(disc_fake_hat))
    
    # Evaluate the real images and the condition with the discriminator.
    disc_real_hat = disc(real, condition)
    
    # Calculate the adversarial loss
    disc_real_loss = adv_criterion(disc_real_hat, torch.ones_like(disc_real_hat))
    
    # Add the two losses
    disc_loss = (disc_fake_loss + disc_real_loss) / 2
    
    return disc_loss
    


#%% U-net Fine Tunning loss

def Unet_FT_Loss(unet,ae,real,condition,criterion_reco,criterion_latent,lambda_reco,lambda_latent):
    '''
    Loss function of the U-net fine tunning according to the paper.

    Parameters
    ----------
    unet : TYPE
        DESCRIPTION.
    ae : TYPE
        DESCRIPTION.
    real : TYPE
        DESCRIPTION.
    condition : TYPE
        DESCRIPTION.
    criterion : TYPE
        DESCRIPTION.
    lambda_reco : TYPE
        DESCRIPTION.
    lambda_latent : TYPE
        DESCRIPTION.

    Returns
    -------
    unet_FT_loss : TYPE
        DESCRIPTION.

    '''
    # Calculate output of U-net
    output = unet(condition)
    
    # Calculate the output of the AE, using the U-net output
    output_latent,output_ae = ae(output)
    
    # output should be the same!
    loss_output = criterion_latent(output,output_ae)
    
    # find the latent space representation of the segementation map
    real_latent,_ = ae(real)
    
    # output and real should be the same in the latent space!
    loss_latent = criterion_latent(output_latent,real_latent)
    
    # loss between segmentation maps
    loss_reco = criterion_reco(output,real)
    
    # final loss
    unet_FT_loss = loss_output + lambda_latent * loss_latent + lambda_reco * loss_reco
    

    return output,unet_FT_loss


#%% GAN Fine Tunning loss

def GAN_FT_Loss(gen,disc,real,condition,adv_criterion,recon_criterion,lambda_reco):
    '''
    

    Parameters
    ----------
    gen : TYPE
        DESCRIPTION.
    disc : TYPE
        DESCRIPTION.
    real : TYPE
        DESCRIPTION.
    condition : TYPE
        DESCRIPTION.
    adv_criterion : TYPE
        DESCRIPTION.
    recon_criterion : TYPE
        DESCRIPTION.
    lambda_reco : TYPE
        DESCRIPTION.

    Returns
    -------
    gen_FT_loss : TYPE
        DESCRIPTION.

    '''

    # create segmentation map
    fake = gen(condition)
    
    # move throw the discriminator 
    disc_fake = disc(fake,condition)
    
    # Calculate the adversarial loss.
    gen_adv_loss = adv_criterion(disc_fake,torch.ones_like(disc_fake))
    
    # Calculate the reconstruction loss.
    loss_reco = recon_criterion(real,fake)
    
    # final loss
    gen_FT_loss = gen_adv_loss + lambda_reco * loss_reco
    
    return fake,gen_FT_loss
















