#!/usr/bin/env python3


from training import Train
import argparse


if __name__ == '__main__':
    
    weight_folder = "./weight/"
    # get the parameters from user
    parser = argparse.ArgumentParser()
    
    
    # wandb
    parser.add_argument("wandb", nargs='?',default= 0, help= "Active W&B service")

    # ResNet50 or ResNet18
    parser.add_argument("network", nargs='?',default = "ResNet18" ,help="ResNet18")
    
    # pretrained or not
    parser.add_argument("pretrain", nargs='?',default = 0,help="Pretrain or not")

    # batch size
    parser.add_argument("batch_size", nargs='?',default= 12, type=int ,help= "batch size")

    # cuda enable
    parser.add_argument("cuda_enable", nargs='?',default= True ,help= "")

    # learning rate
    parser.add_argument("learning_rate", nargs='?',default=0.001, type = float,help= "learning_rate, default is 0.003 ")
    
    # epochs
    parser.add_argument("epochs", nargs='?',default = 10, type = int,help= "epochs, default is 10 ")
    
    # test
    parser.add_argument("test", nargs='?',default = 1 , help= "Testing Model ?")
    
    # Pretrained weight
    parser.add_argument("modelname", nargs='?',default = weight_folder + "with_pretrain_ResNet18_10" + ".pkl" , help= "Pretrained weight")
    
    # load last time model
    parser.add_argument("load", nargs='?',default = 1 , help= "Do you want to load previous model?")
    
    # load model
    parser.add_argument("pretrained", nargs='?',default = 1 )
    
    args = parser.parse_args()
    
    trainer = Train(args)


