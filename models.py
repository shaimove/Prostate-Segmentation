from collections import OrderedDict
from cytoolz.itertoolz import concat, sliding_window
import functools
import monai
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from typing import Callable, Iterable, Optional, Tuple, List

#%% Blocks for U-net

class ConvBlock(nn.Module):
    '''
    ConvBlock Class: perform two Conv2D-Batch Normalization-Activation followed 
    by MaxPooling, with possible drpout layer after the Batch Normalization
    
    Parameters
    ----------
    input_channels : integer
            the number of channels to expect from a given input
    use_dropout : Boolean
        Whether to use Dropout 
        
    Returns
    -------
    None.
    '''

    def __init__(self, input_channels, use_dropout=False):
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(input_channels * 2)
        
        self.conv2 = nn.Conv2d(input_channels * 2, input_channels * 2, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(input_channels * 2)
        
        self.activation = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        if use_dropout: 
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self, x):
        '''
        x: image tensor of shape (batch size, channels, height, width)
        '''
        x = self.conv1(x)
        x = self.batchnorm(x)            
        if self.use_dropout: 
            x = self.dropout(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.batchnorm(x)            
        if self.use_dropout: 
            x = self.dropout(x)
        x = self.activation(x)
        
        x = self.maxpool(x)
        
        return x

class UPConvBlock(nn.Module):
    '''
    UPConvBlock Class:
    Performs an upsampling, a convolution, a concatenation of its two inputs,
    followed by two more convolutions with optional dropout
    input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels, use_dropout=False):
        super(UPConvBlock, self).__init__()
        
        self.upsample = nn.ConvTranspose2d(input_channels, input_channels, kernel_size=2,stride=2)
        
        self.conv1 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=2)
        self.batchnorm1 = nn.BatchNorm2d(input_channels // 2)
        
        self.conv2 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(input_channels // 2)
        
        self.conv3 = nn.Conv2d(input_channels // 2, input_channels // 2, kernel_size=2, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(input_channels // 2)
        
        self.activation = nn.ReLU()
        
        if use_dropout: self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self, x, skip_con_x):
        '''
        x: image tensor of shape (batch size, channels, height, width)
        skip_con_x: the image tensor from the contracting path (from the opposing block of x)
                    for the skip connection
        '''
        x = self.upsample(x)
        x = self.conv1(x)
        skip_con_x = self.crop(skip_con_x, x)
        x = torch.cat([x, skip_con_x], axis=1)
        
        x = self.conv2(x)
        x = self.batchnorm1(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        
        x = self.conv3(x)
        x = self.batchnorm3(x)
        if self.use_dropout: 
            x = self.dropout(x)
        x = self.activation(x)
        
        return x


    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


#%% U-net

class UNet(nn.Module):
    '''
    UNet Class
    A series of 4 contracting blocks followed by 4 expanding blocks to 
    transform an input image into the corresponding paired image, with an upfeature
    layer at the start and a downfeature layer at the end.
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''
    def __init__(self, input_channels, output_channels, hidden_channels=32):
        super(UNet, self).__init__()
        
        self.conv_first = nn.Conv2d(input_channels, hidden_channels, kernel_size=1)
        
        self.contract1 = ConvBlock(hidden_channels, use_dropout=True)
        self.contract2 = ConvBlock(hidden_channels * 2, use_dropout=True)
        self.contract3 = ConvBlock(hidden_channels * 4, use_dropout=True)
        self.contract4 = ConvBlock(hidden_channels * 8)
        self.contract5 = ConvBlock(hidden_channels * 16)
        self.contract6 = ConvBlock(hidden_channels * 32)
        
        # at the bridge we have 2048 channels, maybe reduce to 1024?
        self.expand0 = UPConvBlock(hidden_channels * 64)
        self.expand1 = UPConvBlock(hidden_channels * 32)
        self.expand2 = UPConvBlock(hidden_channels * 16)
        self.expand3 = UPConvBlock(hidden_channels * 8)
        self.expand4 = UPConvBlock(hidden_channels * 4)
        self.expand5 = UPConvBlock(hidden_channels * 2)
        
        self.conv_final = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        '''
        x: image tensor of shape (batch size, channels, height, width)
        '''
        x0 = self.conv_first(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        x5 = self.contract5(x4)
        x6 = self.contract6(x5)
        x7 = self.expand0(x6, x5)
        x8 = self.expand1(x7, x4)
        x9 = self.expand2(x8, x3)
        x10 = self.expand3(x9, x2)
        x11 = self.expand4(x10, x1)
        x12 = self.expand5(x11, x0)
        xn = self.conv_final(x12)
        x_final = self.sigmoid(xn)
        
        return x_final

#%% Discriminator for pix2pix , instead of NLayerDiscriminator

class Discriminator(nn.Module):
    '''
    Discriminator Class
    Structured like the contracting path of the U-Net, the discriminator will
    output a matrix of values classifying corresponding portions of the image as real or fake. 
    Parameters:
        input_channels: the number of image input channels
        hidden_channels: the initial number of discriminator convolutional filters
    '''
    def __init__(self, input_channels, hidden_channels=8):
        super(Discriminator, self).__init__()
        self.conv_first = nn.Conv2d(input_channels, hidden_channels, kernel_size=1)
        self.contract1 = ConvBlock(hidden_channels)
        self.contract2 = ConvBlock(hidden_channels * 2)
        self.contract3 = ConvBlock(hidden_channels * 4)
        self.contract4 = ConvBlock(hidden_channels * 8)
        self.final = nn.Conv2d(hidden_channels * 16, 1, kernel_size=1)

    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        x0 = self.conv_first(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        xn = self.final(x4)
        return xn

#%%

class ConditionalGAN(nn.Module):
    def __init__(self, 
            gen_channels = (16, 32, 64, 128, 256),
            gen_in_channels = 1,
            gen_out_channels = 1,
            dis_channels = (16, 32, 64, 128, 256),
            dis_input_shape = (1, 256, 256),
         ):
        
        self.Generator = monai.UNet(
            dimensions = 2,
            in_channels = gen_in_channels,
            out_channels = gen_out_channels,
            channels = gen_channels,
            strides = 1, 
            kernel_size=3, 
            up_kernel_size=3, 
            num_res_units=2,
            act='PRELU', 
            norm='INSTANCE', 
            dropout=0.0
        )
        
        self.Discriminator = monai.Discriminator(
            in_shape = dis_input_shape, 
            channels = dis_channels, 
            strides = 1, 
            kernel_size=3, 
            num_res_units=2, 
            act='PRELU', 
            norm='INSTANCE', 
            dropout=0.25, 
            bias=True, 
            last_act='SIGMOID'
        )
    
    def get_generator_params(self,):
        return self.Generator.parameters()
        
    def get_discriminator_params(self,):
        return self.Discriminator.parameters()
        
        
    def forward(self, x):
        return self.Generator(x)



#%%

### Code from https://github.com/ShayanPersonal/stacked-autoencoder-pytorch


class CDAutoEncoder(nn.Module):
    r"""
    Convolutional denoising autoencoder layer for stacked autoencoders.
    This module is automatically trained when in model.training is True.
    Args:
        input_size: The number of features in the input
        output_size: The number of features to output
        stride: Stride of the convolutional layers.
    """
    def __init__(self, input_size, output_size):
        super(CDAutoEncoder, self).__init__()

        ### YN: conv layers changed to double conv layers. transposed conv
        ###     changed to upsampling and interpolation.

        self.forward_pass = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(input_size, output_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.backward_pass = nn.Sequential(
            nn.Upsample(
                scale_factor = 2,
                mode = 'trilinear',
            ),
            nn.Conv2d(input_size, output_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(input_size, output_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        ### YN: training will be handled by a container class.
        # self.criterion = nn.MSELoss()
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

    def forward(self, x):
        
        ### YN: training separation and noise addition will be handled by a 
        ###     container class.
        # # Train each autoencoder individually
        # x = x.detach()
        # # Add noise, but use the original lossless input as the target.
        # x_noisy = x * (Variable(x.data.new(x.size()).normal_(0, 0.1)) > -.1).type_as(x)
        # y = self.forward_pass(x_noisy)
        
        y = self.forward_pass(x)

        ### YN: training will be handled by a container class.
        # if self.training:
        #     x_reconstruct = self.backward_pass(y)
        #     loss = self.criterion(x_reconstruct, Variable(x.data, requires_grad=False))
        #     self.optimizer.zero_grad()
        #     loss.backward()
        #     self.optimizer.step()
        
        ### YN: training will be handled by a container class.
        # return y.detach()
        return y

    def reconstruct(self, x):
        return self.backward_pass(x)


class StackedAutoEncoder(nn.Module):
    r"""
    A stacked autoencoder made from the convolutional denoising autoencoders above.
    Each autoencoder is trained independently and at the same time.
    """

    def __init__(self, embedded_dimentions = [3, 8, 16, 32],):
        ### YN: deafult architecture results in 24k params. almost 12k as 
        ### reported in the paper "Learning and Incorporating Shape Models
        ### for Semantic Segmentation"
        super(StackedAutoEncoder, self).__init__()
        
        self.module_names = []
        for i in range(len(embedded_dimentions)-1):
            self.module_names.append(f'CDAE{i}')
            self.add_module(
                f'CDAE{i}',
                CDAutoEncoder(embedded_dimentions[i], embedded_dimentions[i+1])
            )
        self.depth = len(embedded_dimentions)-1
        
    def set_depth(self, depth):
        self.depth = depth
        
    def get_sub_ae_params(self, depth):
        return self._modules[self.module_names[depth]].parameters()

    def forward(self, x, depth = None):
        if depth is None:
            depth = self.depth
        
        embedded = x
        for i in range(depth):
            embedded = self._modules[self.module_names[i]](embedded)
            
        reconstructed = embedded
        for i in range(depth):
            reconstructed = self._modules[self.module_names[depth-i-1]].reconstruct(reconstructed)
        
        return reconstructed
    
    


















