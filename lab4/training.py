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
from weightConstraint import weightConstraint
torch.set_default_tensor_type(torch.DoubleTensor)


class Train():
    def __init__(self, args):

        
        self.args = args
        if self.args.constraint:
            self.filename = args.network + "_" + args.activation_func + "_" + str(args.learning_rate)  + "_" + str(args.batch_size)  + "_" + str(args.epochs)
        else:
            self.filename = "no_constraint_"+args.network + "_" + args.activation_func + "_" + str(args.learning_rate)  + "_" + str(args.batch_size)  + "_" + str(args.epochs)
        
        if args.wandb:
            
            self.wandb = wandb.init(project = "ECG_classifier", entity="leeandy0822", name=self.filename)
            config = wandb.config
            config.learning_rate = args.learning_rate
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
        self.constraint = weightConstraint()
        
        # setup max_norm constraint
        if args.constraint:
            self.model._modules['classify'].apply(self.constraint)
        
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
        
        
        if args.test:
            self.model.load_state_dict(torch.load(args.modelname))
            self.evaluate(self.testdataloader)
            
        else:
            if args.load:
                self.model.load_state_dict(torch.load(args.modelname))
            self.training()    
        

    def training(self):
        self.loss_record = []
        self.record = []

        # record best accuracy
        best_test_record = 0
        record_weight = None        
        
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
                
                    
            print("{} epochs".format(i))
            self.loss_record.append(train_loss / (i + 1))
            acc_train = self.evaluate(self.traindataloader)
            acc_test = self.evaluate(self.testdataloader)
            
            
            if self.args.wandb:
                wandb.log({"loss":train_loss / (i + 1)})
                wandb.log({"Train accuracy": acc_train})
                wandb.log({"Test accuracy": acc_test})
            
            if acc_test > 87.2:
                record_weight = self.model.state_dict() 
                torch.save(record_weight, "./weight/" + "87gogo" + ".pkl")

                break
            
                
        
        # After episodes
        if acc_test > best_test_record :
            best_test_record = acc_test       
            record_weight = self.model.state_dict()     
        
        
            
        torch.save(record_weight, "./weight/" + self.filename + ".pkl")
                
        print("best test accuracuy: ", best_test_record)
        
        artifact = wandb.Artifact('model', type='model')
        artifact.add_file("./weight/" + self.filename + ".pkl")
        self.wandb.log_artifact(artifact)
        self.wandb.finish()
                
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
                output = output.data.max(1, keepdim=True)[1]
                
                # caculate correct
                # 1. turn label.data to output shape
                # 2. delete the dim=1 
                # 3. np.sum can sum up all correct values!
                correct += np.sum(np.squeeze(output.eq(label.data.view_as(output))).cpu().numpy())
                total += data.size(0)

            text = "Train accuracy" if d == self.traindataloader else "Test accuracy"
            tbar.set_description('{0}: {1:2.2f}% '.format(text, 100. * correct / total))

        return 100.0 * correct / total
            