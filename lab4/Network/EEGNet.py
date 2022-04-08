#!/usr/bin/env python3

import torch.nn as nn

# inherit from torch.nn
class EEGNet(nn.Module):
    def __init__(self, activation_func):
        
        activation = {
            'RELU' : nn.RELU(),
            'LeakyRELU' : nn.LeakyReLU(),
            'ELU' : nn.ELU()
        }
        
        self.firstconv = nn.Sequential(
            # because the kernal size is different, so we choose different padding in height and width
            # Height and Width is different!
            nn.Conv2d(1,16, kernal_size=(1,51), stride = (1, 1), padding=(0,25), bias=False),
            nn.BatchNorm2d(16, affine=True)
        )
        
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16,32,kernal_size=(2,1),stride=(1,1),groups=16,bias=False),
            nn.BatchNorm2d(32, affine=True),
            activation[activation_func],
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1 , 4), padding=0),
            nn.Dropout(p=0.25)
        )
        
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1,1), padding=(0,7), bias=False),
            # learn gamma and beta 
            nn.BatchNorm2d(32, affine=True),
            activation[activation_func],
            nn.AvgPool2d(1,8),
            nn.Dropout(p=0.25)
        )
        
        self.classify = nn.Sequential(
            nn.Linear(736, 2, bias=True)
        )
    
    def forward(self,input):
        
        h1 = self.firstconv(input)
        h2 = self.depthwiseConv(h1)
        h3 = self.separableConv(h2)
        output = h3.view(h3.size(0), -1)
        return self.classify(output)
        