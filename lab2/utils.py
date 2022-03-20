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
    
    def tanh (self,x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    def tanh_d(self,x):
        return 1 - self.tanh(x)^2
    
    def relu(self,x):
        if x < 0: return 0
        else: return x
        
    def relu_d(self,x):
        if x < 0: return 0
        else: 1
        
    def none(self,x):
        return x
        
    def none_d(self,x):
        return 1