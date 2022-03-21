#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt

from NeuralNetwork import NeuralNetwork

def generate_linear(n=100):
  
  point = np.random.uniform(0, 1, (n, 2))
  input = []
  label = []
  for p in point:
    input.append([p[0], p[1]])
    if p[0] > p[1]:
      label.append(0)
    else:
      label.append(1)
  return np.array(input), np.array(label).reshape(n, 1)

def generate_XOR_easy():
  input = []
  label = []

  for i in range(11):
    input.append([0.1 * i, 0.1 * i])
    label.append(0)

    if 0.1 * i == 0.5:
      continue

    input.append([0.1 * i, 1 - 0.1 * i])
    label.append(1)

  return np.array(input), np.array(label).reshape(21, 1)

def show_result(x, y, pre_y):
    x = np.transpose(x)
    y = np.transpose(y)
    pre_y = np.transpose(pre_y)
    plt.subplot(2, 1, 1)
    plt.title("Ground Truth", fontsize = 15)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], "ro")
        else:
            plt.plot(x[i][0], x[i][1], "bo")

    plt.subplot(2, 1, 2)
    plt.title("Prediction", fontsize = 15)
    for i in range(x.shape[0]):
        if pre_y[i] <= 0.3:
            plt.plot(x[i][0], x[i][1], "ro")
        else:
            plt.plot(x[i][0], x[i][1], "bo")
  
    plt.show()

if __name__ == "__main__":
    
    # get the parameters from user
    parser = argparse.ArgumentParser()
    parser.add_argument("architecture", nargs='?',default = [2 , 5, 5 ,1],type = list,help="A list to define nn structue, default is [2, 5, 5 ,1]")
    parser.add_argument("activation_func",  nargs='?',default = "tanh", help = "activation function(sigmoid, tanh, none), default is sigmoid")
    parser.add_argument("optimizer", nargs='?',default = "SGD",help = "optimizer function, default is SGD")
    parser.add_argument("data", nargs='?',default="linear",help= "linear or XOR datatype")
    args = parser.parse_args()
    
    classifier = NeuralNetwork(args)
    
    if (args.data == "linear"):
          
        x,y = generate_linear()
        x = np.transpose(x)
        y = np.transpose(y)
        
        classifier.train(x,y)
        classifier.test(x,y)
        
        x,y = generate_linear()
        x = np.transpose(x)
        y = np.transpose(y)
        prediction = classifier.predict(x)
        show_result(x,y,prediction)
        
    if (args.data == "XOR"):
          
        x,y = generate_XOR_easy()
        x = np.transpose(x)
        y = np.transpose(y)      
        classifier.train(x,y)
        classifier.test(x,y)
        prediction = classifier.predict(x)
        show_result(x,y,prediction)


