#!/usr/bin/env python3

import enum
from matplotlib import projections
from torch.autograd import Variable
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from dataset.dataloader import ECGDataset
from Network.DeepConvNet import DeepConvNet
from Network.EEGNet import EEGNet
from tqdm import tqdm
import wandb
torch.set_default_tensor_type(torch.DoubleTensor)


class Train():
    def __init__(self, args):

        
        self.args = args
               
        if args.wandb:
            
            self.wandb = wandb.init(project = "ECG_classifier", entity="leeandy0822")
            config = wandb.config
            config.learning_eate = args.learning_rate
            config.batch_size = args.batch_size
            config.epochs = args.epochs
        

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
        
        self.model = self.model.double()
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
        self.loss_record = []
        self.record = []
        
        if self.args.wandb:
            wandb.watch(self.model)
        
        for i in range(self.args.epochs):
            train_loss = 0.0
            tbar = tqdm(self.traindataloader)
            
            # turn into training mode
            self.model.train()
            for j, (x, y) in enumerate(tbar):
                
                b_x = Variable(x)
                b_y = Variable(y)
                
                if self.args.cuda_enable:
                    b_x = b_x.cuda()
                    b_y = b_y.cuda()
                    
                output = self.model(b_x)
                loss = self.loss(output, b_y.long())
                
                
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                self.optimizer.zero_grad()
                
                tbar.set_description('Train loss: {0:.6f}'.format(train_loss / (j + 1)))


            self.loss_record.append(train_loss / (i + 1))
            acc_train = self.evaluate(self.traindataloader)
            acc_test = self.evaluate(self.testdataloader)
            
            if self.args.wandb:
                wandb.log({"loss":train_loss / (i + 1)})
                wandb.log({"Train accuracy": acc_train})
                wandb.log({"Test accuracy": acc_test})
        

                
    # evaluation
    def evaluate(self, d):

        correct = 0.0
        total = 0.0
        tbar = tqdm(d)
        # turn into evaluation mode
        self.model.eval()
        
        for i, (data, label) in enumerate(tbar):
            
            data, label = Variable(data),Variable(label)
            
            # using cuda
            if self.args.cuda_enable:
                data, label = data.cuda(), label.cuda()
                
            with torch.no_grad():
                output = self.model(data)
                pred = output.data.max(1, keepdim=True)[1]
                correct += np.sum(np.squeeze(pred.eq(label.data.view_as(pred))).cpu().numpy())
                total += data.size(0)

            text = "Train accuracy" if d == self.traindataloader else "Test accuracy"
            tbar.set_description('{0}: {1:2.2f}% '.format(text, 100. * correct / total))

        return 100.0 * correct / total
            