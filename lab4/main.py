#!/usr/bin/env python3

from dataset.dataloader import ECGDataset
from training import Train
import argparse


if __name__ == '__main__':
    
    # get the parameters from user
    parser = argparse.ArgumentParser()
    
    
    # wandb
    parser.add_argument("wandb", nargs='?',default= False,help= "Active W&B service")


    parser.add_argument("network", nargs='?',default = "DeepConvNet",help="A list to define nn structue, default is [2, 5, 5 ,1]")
    # activation function
    parser.add_argument("activation_func",  nargs='?',default = "ELU", help = "")

    # optimizer
    parser.add_argument("optimizer", nargs='?',default = "Adam",help = "optimizer function, default is SGD")
    
    # batch size
    parser.add_argument("batch_size", nargs='?',default= 16 , type=int ,help= "batch size")

    # cuda enable
    parser.add_argument("cuda_enable", nargs='?',default= False ,help= "")
    
    # loss function 
    parser.add_argument("loss_function", nargs='?',default="LMS",help= "CE (cross entropy)or LMS ")
    # learning rate
    parser.add_argument("learning_rate", nargs='?',default=0.0005, type = float,help= "learning_rate, default is 0.01 ")
    # epochs
    parser.add_argument("epochs", nargs='?',default=200, type = int,help= "epochs, default is 10000 ")

    
    args = parser.parse_args()
    
    
    trainer = Train(args)
    trainer.training()


