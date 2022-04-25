#!/usr/bin/env python3 

import torch.nn as nn
from torchvision import models

class ResNet18(nn.Module):
    
    def __init__(self, num_class, pretrained):
        super(ResNet18,self).__init__()

        self.model = models.resnet18(pretrained=pretrained)
        
        # last feature output
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 256)
        self.activate = nn.LeakyReLU()
        self.fc = nn.Linear(256, num_class)
    
    def forward(self, x):
        x = self.model(x)
        x = self.activate(x)
        x = self.fc(x)
        return x
    
    
class ResNet50(nn.Module):
    
    def __init__(self, num_class, pretrained):
        super(ResNet50,self).__init__()

        self.model = models.resnet50(pretrained=pretrained)        
        # last feature output
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 512)
        self.activate = nn.LeakyReLU()
        self.fc = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_class)
    
    def forward(self, x):
        x = self.model(x)
        x = self.activate(x)
        x = self.fc(x)
        x = self.activate(x)
        x = self.fc2(x)
        return x
    
    
    
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False