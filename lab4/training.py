#!/usr/bin/env python3

import enum
from tkinter import Variable
from gpg import Data
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from dataset.dataloader import ECGDataset
from Network.DeepConvNet import DeepConvNet
from Network.EEGNet import EEGNet
from tqdm import tqdm


class Train():
    def __init__(self, args):
        
        self.args = args
        # args.network
        # args.
        
        # get dataset 
        self.train_set = ECGDataset("train")
        self.test_set = ECGDataset("test")
        
        # define network 
        if args.network == "DeepConvNet":
            self.model = DeepConvNet(args.activation_func)
        elif args.network == "EEGNet":
            self.model = EEGNet(args.activation_func)
            
        print(self.model)    
        
        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = args.learning_rate)
        
        # loss
        self.loss = nn.CrossEntropyLoss()
        
        # create dataloader 
        self.traindataloader = DataLoader(dataset = self.train_set, batch_size= args.batch_size, shuffle = True)
        self.testdataloader =  DataLoader(dataset = self.test_set, batch_size= args.batch_size, shuffle = True)
        
        # use cuda
        if args.cuda_enable:
            self.model = self.model.cuda()
            self.loss = self.loss.cuda()
            print("Using GPU Acceleration!!")
        
    def training(self):
        
        for i in range(self.args.epochs):
            train_loss = 0.0
            tbar = tqdm(self.traindataloader)
            self.model.train()
            
            for j, (x, y) in enumerate(self.traindataloader):
                b_x = x
                b_y = y
                
                if self.args.cuda_enable:
                    b_x = b_x.cuda()
                    b_y = b_y.cuda()
                    
                output = self.model(b_x)
                loss = self.loss(output, b_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                self.optimizer.zero_grad()
                
                tbar.set_description('Train loss: {0:.6f}'.format(train_loss / (j + 1)))

                
                
                
                
                
            