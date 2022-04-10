#!/usr/bin/env python3

import torch.nn as nn

# inherit from torch.nn
class DeepConvNet(nn.Module):
    def __init__(self, activation_func):
        super(DeepConvNet, self).__init__()
        activation = {
            'ReLU' : nn.ReLU(),
            'LeakyReLU' : nn.LeakyReLU(),
            'ELU' : nn.ELU()
        }
        
        self.featurelayer1 = nn.Sequential(
            # parameters = Cin*h*w*Cout + Cout(bias)
            # https://towardsdatascience.com/pytorch-conv2d-weights-explained-ff7f68f652eb
            nn.Conv2d(1,25, kernel_size=(1,5), stride = (1, 2), bias=True),
            nn.Conv2d(25,25,kernel_size=(2,1), bias=True),
            # update gamma and beta parameters = 2x25
            nn.BatchNorm2d(25, affine=True),
            activation[activation_func],
            # kernel size, stride
            nn.MaxPool2d(1,2),
            nn.Dropout(0.5)
            
        )
        
        self.featurelayer2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size = (1,5), stride = (1,2), bias = True),
            nn.BatchNorm2d(50, affine = True),
            activation[activation_func],
            nn.MaxPool2d((1,2)),
            nn.Dropout(0.5)
        )
        
        self.featurelayer3 = nn.Sequential(
            nn.Conv2d(50,100,kernel_size=(1,5), bias = True),
            nn.BatchNorm2d(100, affine=True),
            activation[activation_func],
            nn.MaxPool2d((1,2)),
            nn.Dropout(0.5)
        )

        self.featurelayer4 = nn.Sequential(
            nn.Conv2d(100,200,kernel_size=(1,5), bias= True),
            nn.BatchNorm2d(200, affine=True),
            activation[activation_func],
            nn.MaxPool2d((1,2)),
            nn.Dropout(0.5)
        )
        
        self.classify = nn.Sequential(
            nn.Linear(1600, 2, bias=True),
            nn.Softmax(dim=1)
        ) 
    
    def forward(self,input):
        
        h1 = self.featurelayer1(input)
        h2 = self.featurelayer2(h1)
        h3 = self.featurelayer3(h2)
        h4 = self.featurelayer4(h3)
        # Flatten
        output = h4.view(h4.size(0), -1)
        return self.classify(output)
        