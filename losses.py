import torch
from torch import nn

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



def get_gen_loss(gen, disc, real, condition, adv_criterion, recon_criterion, lambda_recon):
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




def get_disc_loss(gen, disc, real, condition, adv_criterion):
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
    
#%% identity loss

def get_identity_loss(real, gen, identity_criterion):
    '''
    Return the identity loss of the generator given inputs
    (and the generated images for testing purposes).
    Parameters:
        real: the real segmentation 
        gen: the generator takes segmentations and returns the images and "try" to fix them
        identity_criterion: the identity loss function; takes the real segmentation and
                        returns the identity loss (which you aim to minimize)
    '''
    identity = gen(real)
    identity_loss = identity_criterion(identity,real)
    
    return identity_loss, identity





















