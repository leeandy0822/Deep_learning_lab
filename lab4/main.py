#!/usr/bin/env python3

from dataset.dataloader import ECGDataset
from training import Train
import argparse


if __name__ == '__main__':
    
    weight_folder = "./weight/"
    # get the parameters from user
    parser = argparse.ArgumentParser()
    
    
    # wandb
    parser.add_argument("wandb", nargs='?',default= 0, help= "Active W&B service")


    parser.add_argument("network", nargs='?',default = "EEGNet",help="DeepConvNet or EEGNet")
    
    # activation function
    parser.add_argument("activation_func",  nargs='?',default = "LeakyReLU", help = "LeakyReLU, ReLU, ELU")
    
    # batch size
    parser.add_argument("batch_size", nargs='?',default= 12, type=int ,help= "batch size")

    # cuda enable
    parser.add_argument("cuda_enable", nargs='?',default= True ,help= "")

    # learning rate
    parser.add_argument("learning_rate", nargs='?',default=0.0002, type = float,help= "learning_rate, default is 0.003 ")
    
    # epochs
    parser.add_argument("epochs", nargs='?',default = 300, type = int,help= "epochs, default is 300 ")
    
    # constraint
    parser.add_argument("constraint", nargs='?',default = True , help= "Last layer constraint")
    
    # test
    parser.add_argument("test", nargs='?',default = 1 , help= "Testing Model ?")
    
    # test
    parser.add_argument("modelname", nargs='?',default = weight_folder + "bestgogogo" , help= "Testing Model ?")
    
    # load model
    parser.add_argument("load", nargs='?',default = 0 )
    
    args = parser.parse_args()
    
    trainer = Train(args)


