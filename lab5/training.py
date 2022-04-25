#!/usr/bin/env python3


from debugpy import configure
from torch.autograd import Variable
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from Network.models import ResNet18, ResNet50
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from dataset.dataloader import RetinopathyLoader
import itertools
import os

class Train():
    def __init__(self, args):

        self.args = args
        self.img_path = "/home/leeandy/Deep_learning_lab/lab5/dataset/data";
        
        if args.pretrained:
            self.filename = "with_pretrain_" + args.network   + "_" + str(args.epochs)
        else:
            self.filename = "without_pretrain_" +args.network  + "_" + str(args.epochs)

        if args.wandb:    
            self.wandb = wandb.init(project = "ResNet_fintune", entity="leeandy0822", name=self.filename)
            config = wandb.config
            config.learning_rate = args.learning_rate
            config.batch_size = args.batch_size
            config.epochs = args.epochs      
            
        
        # data
        self.train_set = RetinopathyLoader("train")
        self.test_set = RetinopathyLoader("test")
        
        # define network 
        if args.network == "ResNet18":
            self.model = ResNet18( 5, args.pretrain)
            
        elif args.network == "ResNet50":
            self.model = ResNet50( 5, args.pretrain)
        
        print(self.model)
        

        # create dataloader 
        self.traindataloader = DataLoader(dataset = self.train_set, batch_size= args.batch_size, num_workers=4)
        self.testdataloader =  DataLoader(dataset = self.test_set, batch_size= args.batch_size, num_workers=4)
        
        # optimizer
        self.optimizer =  torch.optim.SGD(self.model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay = 5e-4)
        
        # loss
        self.loss = nn.CrossEntropyLoss()
        
        # use cuda
        if args.cuda_enable:
            self.model = self.model.cuda()
            self.loss = self.loss.cuda()
            print("Using GPU Acceleration!!")

        
        if args.test:
            self.model.load_state_dict(torch.load(args.modelname))
            self.evaluate(self.testdataloader)
            self.plot_confusion_matrix(5)
            
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
                
                self.optimizer.zero_grad()
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
            
            
            
            
    def plot_confusion_matrix(self, num_class):
        
        print("\n Plot Confusion Matrix... \n")
        confusion_matrix = np.zeros((num_class,num_class))
        # set model to be evaluation mode
        self.model.eval()
        
        for i, (data, label) in enumerate(self.testdataloader):
            data, label = Variable(data),Variable(label)
                            
            # using cuda
            if self.args.cuda_enable:
                data, label = data.cuda(), label.cuda()
                
            with torch.no_grad():
                prediction = self.model(data)
                pred = prediction.data.max(1, keepdim=True)[1]

                ground_truth = pred.cpu().numpy().flatten()
                actual = label.cpu().numpy().astype('int')

                for j in range(len(ground_truth)):
                    confusion_matrix[actual[j]][ground_truth[j]] += 1

        # normalization
        for i in range(num_class):
            confusion_matrix[i,:] /=  sum(confusion_matrix[i,:])

        plt.figure(1)
        plt.imshow(confusion_matrix, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar()

        thresh = np.max(confusion_matrix) / 2.
        for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
            plt.text(j, i, format(confusion_matrix[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

        tick_marks = np.arange(num_class)
        plt.xticks(tick_marks)
        plt.yticks(tick_marks)

        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        plt.savefig(os.path.join("./plot", self.filename + '_confusion_matrix.png'))