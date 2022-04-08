import numpy as np

import torch
from  torch.utils.data import Dataset, dataloader

class ECGDataset(Dataset):
    
    def __init__(self, arg):
        
        self.arg = arg
        self.data, self.label = self.read_bci_data();
        
    def __getitem__(self, index):
        
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


    def read_bci_data(self):
        S4b_train = np.load('./dataset/S4b_train.npz')
        X11b_train = np.load('./dataset/X11b_train.npz')
        S4b_test = np.load('./dataset/S4b_test.npz')
        X11b_test = np.load('./dataset/X11b_test.npz')

        train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
        train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
        test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
        test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)

        train_label = train_label - 1
        test_label = test_label -1
        train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
        test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

        mask = np.where(np.isnan(train_data))
        train_data[mask] = np.nanmean(train_data)

        mask = np.where(np.isnan(test_data))
        test_data[mask] = np.nanmean(test_data)


        if self.arg == "train":
            
            print("training set shape: ",train_data.shape, train_label.shape)
            return train_data, train_label

        elif self.arg == "test":
            
            print("testing set shape: ",test_data.shape, test_label.shape)
            return test_data, test_label
