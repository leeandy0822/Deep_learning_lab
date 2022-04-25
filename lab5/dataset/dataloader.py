#!/usr/bin/env python3
import pandas as pd
import numpy as np
from PIL import Image
import os
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import gdown
from zipfile import ZipFile
import shutil

class RetinopathyLoader(Dataset):
    def __init__(self, mode):
        """
        Args:
            mode : Indicate procedure status(training or testing)
            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = os.path.join("./dataset/data")
        self.img_name, self.label = self.__getData(mode)
        self.transform = self.__transform(mode)
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""

        return len(self.img_name)

    def __getitem__(self, index):

        # read rgb images
        rgb_image = Image.open(os.path.join(self.root, self.img_name[index] + ".jpeg")).convert('RGB')

        # transformate pics
        img = self.transform(rgb_image)
        label = self.label[index]

        return img, label


    def __transform(self, mode):
        
        if mode == "train":
            # data augmentation
            transform = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # pytorch common mean and std
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        else:
            # we do not need to flip in evaluation step
            transform = transforms.Compose([
            transforms.ToTensor(),
            # pytorch common mean and std
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        return transform

    def __getData(self, mode):

        if mode == 'train':
            img = pd.read_csv('./dataset/train_img.csv', header = None)
            label = pd.read_csv('./dataset/train_label.csv', header = None)
            return np.squeeze(img.values), np.squeeze(label.values)
        elif mode == "test":
            img = pd.read_csv('./dataset/test_img.csv', header = None)
            label = pd.read_csv('./dataset/test_label.csv', header = None)
            return np.squeeze(img.values), np.squeeze(label.values)
        else:
            raise Exception("Error! Please input train or test")
