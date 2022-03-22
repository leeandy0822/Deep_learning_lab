#!/usr/bin/env python3


import numpy as np
import matplotlib.pyplot as plt

import math
from utils import activate_func

class NeuralNetwork(object ):
    
    def __init__(self,args):
        
        # args should pass architecture, activation_func, optimizaer
        
        # define nn structure
        self.n = np.array(args.architecture)
        self.L = self.n.size - 1
        
        self.input_layer = self.n[0]
        self.output_layer = self.n[-1]
        
        self.parameters = {}

        # define the hperparameter
        self.learning_rate = args.learning_rate
        self.epochs = args.epochs
        self.batch = 10
        
        
        # setup the weights
        for i in range(1, self.L +1):
            self.parameters['W' + str(i)] = np.random.randn(self.n[i], self.n[i-1])
            self.parameters['a' + str(i)] = np.ones((self.n[i], 1))
            self.parameters['z' + str(i)] = np.ones((self.n[i], 1))
        
        self.parameters['a0'] = np.ones((self.n[i],1))
        self.parameters['Cost'] = 1
        self.derivatives = {}
        
        # setup activation function
        self.activation_name = args.activation_func
        self.activation, self.activation_back = activate_func(args.activation_func).setup()
        self.loss_name = args.loss_function

        
        # define optimizer: SGD, Momentem, Adagrad, Adams
        self.optimizer = args.optimizer
        
        self.momentum = 0.9
        self.vel_W1 = 0
        self.vel_W2 = 0
        self.vel_W3 = 0
    
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

        al = self.parameters['a' + str(self.L)]
        zl = self.parameters['z' + str(self.L)]

        if self.loss_name == "LMS":
            self.derivatives['dz' + str(self.L)] = 2*(al - y) * derivative_activation(zl)
        else:
            self.derivatives['dz' + str(self.L)] = -((y*(1-al) - (1-y)*al)/al*(1-al)) * derivative_activation(zl)
        
        
        self.derivatives['dW' + str(self.L)] = np.dot(self.derivatives['dz' + str(self.L)], np.transpose(self.parameters['a' + str(self.L - 1)]))
        
        for i in range(self.L-1, 0 , -1):
            
            self.derivatives['dz' + str(i)] =  np.dot(np.transpose(self.parameters['W' + str(i + 1)]), 
                                                      self.derivatives['dz' + str(i + 1)])*derivative_activation(self.parameters['z' + str(i)])
            
            self.derivatives['dW' + str(i)] =  np.dot(self.derivatives['dz' + str(i)], np.transpose(self.parameters['a' + str(i - 1)]))
            
    
    
    def loss(self,y):
        result = self.parameters['a' + str(self.L)] 
        # cross_entropy loss function will cause nan in cost
        if   self.loss_name == "LMS":
            self.parameters['Cost'] = np.mean((result - y) ** 2)
            
        else:
            self.parameters['Cost'] = np.mean(-(y*np.log(result) + 
                                 (1-y)*np.log( 1 - result)))

            
    def update_weight(self, epochs):
        
        if (self.optimizer == "SGD"):
            
            for i in range(1,self.L+1):
                self.parameters['W' + str(i)] -= self.learning_rate * self.derivatives['dW' + str(i)]
                
        if (self.optimizer == "momentum"):
            
            self.vel_W1 = self.momentum*self.vel_W1 + self.learning_rate*self.derivatives['dW1']
            self.vel_W2 = self.momentum*self.vel_W2 + self.learning_rate*self.derivatives['dW2']
            self.vel_W3 = self.momentum*self.vel_W3 + self.learning_rate*self.derivatives['dW3']
            self.parameters['W1'] -= self.vel_W1
            self.parameters['W2'] -= self.vel_W2
            self.parameters['W3'] -= self.vel_W3
            
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
            loss = self.parameters['Cost']
            loss_record.append(loss)
            
            if (i+1) % 50 == 1:
            
                print("loss of {} ".format(i+1)," epoch : {} ".format(loss))

        plt.title("learning curve", fontsize = 18)
        ep = [x for x in range(1, self.epochs + 1)]
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.plot(ep, loss_record)
        plt.show()

        
    def test(self,x,y):
        # predict  
        y_predict = self.predict(x)
        correct_predict = 0
        
        for j in range(len(y_predict[0])):
            if y_predict[:,j]  <= 0.8:
                y_predict[:,j] = 0
            else:
                y_predict[:,j] = 1
                
            if y_predict[:,j] ==  y[:,j] :
                correct_predict+=1
        
        print("accuracy: {} %".format(correct_predict/len(y_predict[0])*100))

            
        
    def predict(self,x):
        self.forward_propagation(x)
        return self.parameters['a' + str(self.L)]