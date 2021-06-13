from collections import OrderedDict
from cytoolz.itertoolz import concat, sliding_window
import functools
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
        self.batchnorm2 = nn.BatchNorm2d(input_channels * 2)
        
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
        x = self.batchnorm1(x)            
        if self.use_dropout: 
            x = self.dropout(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.batchnorm2(x)            
        if self.use_dropout: 
            x = self.dropout(x)
        x = self.activation(x)
        
        x = self.maxpool(x)
        
        return x
#%% UpConv for pix2pix

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


    def crop(self, tensor_to_crop, x):
        _, _, H, W = x.shape
        tensor_to_crop   = torchvision.transforms.CenterCrop([H, W])(tensor_to_crop)
        return tensor_to_crop


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
    def __init__(self, input_channels, output_channels, hidden_channels=16):
        super(UNet, self).__init__()
        
        # start squence
        self.conv_first = nn.Conv2d(input_channels, hidden_channels, kernel_size=1)
        
        self.conv1 = ConvBlock(hidden_channels)
        self.conv2 = ConvBlock(hidden_channels * 2)
        self.conv3 = ConvBlock(hidden_channels * 4)
        self.conv4 = ConvBlock(hidden_channels * 8)
        self.conv5 = ConvBlock(hidden_channels * 16)
        self.conv6 = ConvBlock(hidden_channels * 32)
        
        self.expand0 = UPConvBlock(hidden_channels * 64)
        self.expand1 = UPConvBlock(hidden_channels * 32)
        self.expand2 = UPConvBlock(hidden_channels * 16)
        self.expand3 = UPConvBlock(hidden_channels * 8)
        self.expand4 = UPConvBlock(hidden_channels * 4)
        self.expand5 = UPConvBlock(hidden_channels * 2)
        
        self.conv_final = nn.Conv2d(hidden_channels, output_channels, kernel_size=1)
        self.sigmoid = torch.nn.Sigmoid()
    
        self._init_weights()
    
    
    def _init_weights(self):
        # initiate with Xavier initialization
        for m in self.modules():
            if type(m) in {nn.Conv2d,nn.ConvTranspose2d}:
                nn.init.xavier_normal_(m.weight) # Weight of layers
                
                if m.bias is not None: 
                    m.bias.data.fill_(0.01)  # if we have bias
                    
            if type(m) in {nn.BatchNorm2d}:
                nn.init.normal_(m.weight) # Weight of layers
                if m.bias is not None:
                    m.bias.data.fill_(0.01)  # if we have bias
                    
    def forward(self, x):
        '''
        x: image tensor of shape (batch size, channels, height, width)
        '''
        x0 = self.conv_first(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        
        x7 = self.expand0(x6, x5)
        x7 = self.expand0(x6, x5)
        x8 = self.expand1(x7, x4)
        x9 = self.expand2(x8, x3)
        x10 = self.expand3(x9, x2)
        x11 = self.expand4(x10, x1)
        x12 = self.expand5(x11, x0)
        
        xn = self.conv_final(x12)
        x_final = self.sigmoid(xn)
        
        return x_final

#%% Discriminator for pix2pix

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
        
        self._init_weights()
    
    def _init_weights(self):
        # initiate with Xavier initialization
        for m in self.modules():
            if type(m) in {nn.Conv2d,nn.ConvTranspose2d}:
                nn.init.xavier_normal_(m.weight) # Weight of layers
                
                if m.bias is not None: 
                    m.bias.data.fill_(0.01)  # if we have bias
                    
            if type(m) in {nn.BatchNorm2d}:
                nn.init.normal_(m.weight) # Weight of layers
                if m.bias is not None:
                    m.bias.data.fill_(0.01)  # if we have bias

    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        x0 = self.conv_first(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        xn = self.final(x4)
        return xn


#%% UpCONV for AE (Auto-encoder)

class UPConvBlockAE(nn.Module):
    '''
    UPConvBlock Class:
    Performs an upsampling, followed by two convolutions
    input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_channels):
        super(UPConvBlockAE, self).__init__()
        
        self.upsample = nn.ConvTranspose2d(input_channels, input_channels, kernel_size=2,stride=2)
        
        self.conv1 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=2)
        self.batchnorm1 = nn.BatchNorm2d(input_channels // 2)
        
        self.conv2 = nn.Conv2d(input_channels // 2, input_channels // 2, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(input_channels // 2)
        
        self.conv3 = nn.Conv2d(input_channels // 2, input_channels // 2, kernel_size=2, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(input_channels // 2)
        
        self.activation = nn.ReLU()

    def forward(self, x):
        '''
        x: image tensor of shape (batch size, channels, height, width)
        skip_con_x: the image tensor from the contracting path (from the opposing block of x)
                    for the skip connection
        '''
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.batchnorm1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        
        x = self.conv3(x)
        x = self.batchnorm3(x)
        
        x = self.activation(x)
        
        return x

    
#%% AE to reconstruct the articale

class AE(nn.Module):
    '''
    AE Class
    A series of 4 contracting blocks followed by 4 expanding blocks to 
    transform an input image into the corresponding paired image, with an upfeature
    layer at the start and a downfeature layer at the end.
    Values:
        input_channels: the number of channels to expect from a given input
        output_channels: the number of channels to expect for a given output
    '''
    def __init__(self, input_channels, output_channels, hidden_channels=16):
        super(AE, self).__init__()
        
        self.conv_first = nn.Conv2d(input_channels, hidden_channels, kernel_size=1)
        
        self.conv1 = ConvBlock(hidden_channels)
        self.conv2 = ConvBlock(hidden_channels * 2)
        self.conv3 = ConvBlock(hidden_channels * 4)
        self.conv4 = ConvBlock(hidden_channels * 8)
        
        self.expand1 = UPConvBlockAE(hidden_channels * 16)
        self.expand2 = UPConvBlockAE(hidden_channels * 8)
        self.expand3 = UPConvBlockAE(hidden_channels * 4)
        self.expand4 = UPConvBlockAE(hidden_channels * 2)
        
        self.conv_final = nn.Conv2d(hidden_channels, output_channels, kernel_size=1)
        self.sigmoid = torch.nn.Sigmoid()
    
        self._init_weights()
    
    
    def _init_weights(self):
        # initiate with Xavier initialization
        for m in self.modules():
            if type(m) in {nn.Conv2d,nn.ConvTranspose2d}:
                nn.init.xavier_normal_(m.weight) # Weight of layers
                
                if m.bias is not None: 
                    m.bias.data.fill_(0.01)  # if we have bias
                    
            if type(m) in {nn.BatchNorm2d}:
                nn.init.normal_(m.weight) # Weight of layers
                if m.bias is not None:
                    m.bias.data.fill_(0.01)  # if we have bias
                    
    def forward(self, x):
        '''
        x: image tensor of shape (batch size, channels, height, width)
        '''
        x0 = self.conv_first(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        Ex = self.conv4(x3)

        x5 = self.expand1(Ex)
        x6 = self.expand2(x5)
        x7 = self.expand3(x6)
        x8 = self.expand4(x7)

        xn = self.conv_final(x8)
        x_final = self.sigmoid(xn)
        
        return Ex,x_final



