#!/usr/bin/env python3


import numpy as np
class activate_func(object):
    
    def __init__(self,args):
        self.args = args
        
    def setup(self):
        
        if self.args == "sigmoid":
            return self.sigmoid, self.sigmoid_d
        
        elif self.args == "tanh":
            return self.tanh, self.tanh_d
        
        elif self.args == "relu":
            return self.relu, self.relu_d
        
        else:
            return self.none, self.none_d
        
    def sigmoid(self,x):

        z = np.exp(-x)
        sig = 1 / (1 + z)
        return sig

        
    def sigmoid_d(self,z):
        
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    # tanh project to 0~1 
    def tanh (self,x):
        return ((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)) + 1)/2

    # the derivative is kind of special, too
    def tanh_d(self,x):
        return 2/((np.exp(x) + np.exp(-x))**2)
    
    def relu(self,x):
        y = np.copy(x)
        y[y<0] = 0
        return y
        
    def relu_d(self,x):
        y = np.copy(x)
        y[y>=0] = 1
        y[y<0] = 0
        return y
    
    def none(self,x):
        
        return abs(x)
        
    def none_d(self,x):
        y = np.zeros(x.shape)
        y[y==0] = 1
        return y