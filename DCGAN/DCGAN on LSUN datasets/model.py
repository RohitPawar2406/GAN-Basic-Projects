# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 15:29:35 2023

@author: Rohit
"""
import torch 
from torch import nn

class Generator(nn.Module):
    def __init__(self, latenDim, featuresGen, outputChannels):
        super(Generator, self).__init__();
        
        self.gen = nn.Sequential(
            # Input -> [B,100,1,1]
            
            # First Layer
            nn.ConvTranspose2d(latenDim, featuresGen*8, 4, 1, 0),
            nn.BatchNorm2d(featuresGen*8),
            nn.ReLU(),
            
            # Second Layer
            nn.ConvTranspose2d(featuresGen*8, featuresGen*4, 4, 2, 1),
            nn.BatchNorm2d(featuresGen*4),
            nn.ReLU(),
            
            # Third layer
            nn.ConvTranspose2d(featuresGen*4, featuresGen*2, 4, 2, 1),
            nn.BatchNorm2d(featuresGen*2),
            nn.ReLU(),
            
            # Fourth Layer
            nn.ConvTranspose2d(featuresGen*2, outputChannels, 4, 2, 1),
            nn.Tanh()
            )
        
    def forward(self, x):
        return self.gen(x)


class Discriminator(nn.Module):
    def __init__(self, imgChannels, featuresDisc) -> None:
        super().__init__()

        self.disc = nn.Sequential(
            # Input -> [B,C, 64, 64]
            nn.Conv2d(imgChannels, featuresDisc, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
