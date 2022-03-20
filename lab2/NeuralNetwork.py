#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("agg")
import math
from utils import activate_func

# This is a two-layer neural network


class NeuralNetwork(object ):
    
    def __init__(self,args):
        
        # args should pass architectur, activation_func, optimizaer
        
        # define nn structure
        self.n = np.array(args.architecture)
        self.L = self.n.size - 1
        
        self.input_layer = self.n[0]
        self.output_layer = self.n[-1]
        
        self.parameters = {}

        # define the hperparameter
        self.learning_rate = 0.1
        self.epochs = 10000
        self.batch = 10
        
        
        # setup the weights
        for i in range(1, self.L +1):
            self.parameters['W' + str(i)] = np.random.randn(self.n[i], self.n[i-1]) * 0.1
            self.parameters['a' + str(i)] = np.ones((self.n[i], 1))
            self.parameters['z' + str(i)] = np.ones((self.n[i], 1))
        
        self.parameters['a0'] = np.ones((self.n[i],1))
        self.parameters['Cost'] = 1
        self.derivatives = {}
        
        # setup activation function
        self.activation, self.activation_back = activate_func(args.activation_func).setup()
        

        
        # define optimizer: SGD, Momentem, Adagrad, Adams
        self.optimizer = args.optimizer
        

    def forward_propagation(self,input):
        
        # define which activation function 
        activation_function = self.activation
        
        # NN forward propagation
        self.parameters['a0'] = input
        
        for l in range(1, self.L + 1):
            self.parameters['z' + str(l)] = np.dot(self.parameters['W' + str(l)], self.parameters['a' + str(l - 1)])
            self.parameters['a' + str(l)] = activation_function(self.parameters['z' + str(l)])
        
    
    def backward_propagation(self, y):
        
        derivative_activation = self.activation_back
        
        self.derivatives['dz' + str(self.L)] = self.parameters['a' + str(self.L)] - y
        self.derivatives['dW' + str(self.L)] = np.dot(self.derivatives['dz' + str(self.L)], np.transpose(self.parameters['a' + str(self.L - 1)]))
        
        for i in range(self.L-1, 0 , -1):
            
            self.derivatives['dz' + str(i)] =  np.dot(np.transpose(self.parameters['W' + str(i + 1)]), 
                                                      self.derivatives['dz' + str(i + 1)])*derivative_activation(self.parameters['z' + str(i)])
            
            self.derivatives['dW' + str(i)] =  np.dot(self.derivatives['dz' + str(i)], np.transpose(self.parameters['a' + str(i - 1)]))
            
    
    
    def loss(self,y):
        self.parameters['Cost'] = -(y*np.log(self.parameters['a' + str(self.L)]) + 
                                 (1-y)*np.log( 1 - self.parameters['a' + str(self.L)]))
        
    def update_weight(self, epochs):
        
        if (self.optimizer == "SGD"):
            
            for i in range(1,self.L+1):
                self.parameters['W' + str(i)] -= self.learning_rate * self.derivatives['dW' + str(i)]

    def split_data(self,x, y , split_percent):
        

        size = x.shape[1]
        split_index = (int)(size* split_percent)
        index = np.linspace(0, size-1, size,dtype= int)
        np.random.shuffle(index)
        
        train_x = x[:,index[:split_index]]
        val_x = x[:,index[split_index :]]
        train_y = y[:,index[:split_index]]
        val_ｙ = y[:,index[split_index :]]
        
        return train_x, val_x, train_y, val_ｙ
        
    def predict(self,x):
        self.forward_propagation(x)
        return self.parameters['a' + str(self.L)]
        
    def train(self, x, y):
        
        train_x, val_x, train_y, val_y = self.split_data(x,y,0.8)
        
        num_iter = self.epochs/self.batch
        loss_record = []
        for i in range(self.epochs):
            
            batch_value = 0
            correct_predict = 0

            
            # there will be epochs/batch iterations to do in loop
            while(batch_value < len(train_x[1])):
                
                x = train_x[:,batch_value:batch_value+self.batch]
                y = train_y[:,batch_value:batch_value+self.batch]

                self.forward_propagation(x)
                self.loss(y)
                self.backward_propagation(y)
                self.update_weight(self.epochs)
                
                batch_value += self.batch
               
            # predict  
            y_predict = self.predict(val_x)

            for j in range(len(y_predict[0])):
                if y_predict[:,j] - val_y[:,j] <= 0.1:
                    correct_predict += 1
                
            self.loss(val_y)
            
            if (i+1)%1 == 0:
                print("epoch : {}".format(i+1))
                loss = np.sum(self.parameters['Cost'])/len(val_y[0])
                print("loss: {}".format(loss))
            
                
                
                
            
        
    def test(self,x,y):
        # predict  
        y_predict = self.predict(x)
        correct_predict = 0
        
        for j in range(len(y_predict[0])):
            if y_predict[:,j] - y[:,j] <= 0.1:
                correct_predict += 1
        
        print("accuracy: {} %".format(correct_predict/len(y_predict[0])*100))

            