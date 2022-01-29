import numpy as np
import math 
import csv
import sys
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1 / (1 + np.exp(-x))
  

class Fully_Connected_Layer:
    def __init__(self, learning_rate):
        self.InputDim = 784
        self.HiddenDim = 128
        self.OutputDim = 10
        self.learning_rate = learning_rate
        self.loss = []
        
        '''Weight Initialization'''
        self.ali = []
        self.W1 = np.random.randn(self.InputDim, self.HiddenDim)
        self.W2 = np.random.randn(self.HiddenDim, self.OutputDim) 
        
    def Forward(self, x_train):
        '''Implement forward propagation'''
        z_ = x_train @ self.W1
        z = sigmoid(z_)
        y_ = z @ self.W2
        o = sigmoid(y_)
        self.ali = [z_, z, y_, o]
        return o
    
    def Backward(self, x_train, y_train):
        '''Implement backward propagation'''
        '''Update parameters using gradient descent'''
        z_, z, y_, o = self.ali
        self.W1 = self.W1 - self.learning_rate/x_train.shape[0] * x_train.T @  ( ( ((o - y_train) * (o * (1- o))) @ self.W2.T) * (z * (1- z))   )
        self.W2 = self.W2 - self.learning_rate/x_train.shape[0] * z.T @ ((o - y_train) * (o * (1- o)))


    def Train(self, Input, Label):
        o = self.Forward(Input)
        self.loss += ((o-Label) @ (o-Label).T).sum()
        self.Backward(Input, Label)        

    def Accuracy(self, x_test, y_test): 
        o = self.Forward(x_test)
        return np.mean(np.argmax(o, axis=1) == np.argmax(y_test, axis=1)) 


learning_rate = 1
'''Construct a fully-connected network'''        
Network = Fully_Connected_Layer(learning_rate)

fileName = sys.argv[1]
data = np.loadtxt(open(fileName, 'rb'), delimiter=',')
train_data = data[:,:784]

tmp = data[:, 784].astype(int)
train_label = np.zeros((tmp.size, tmp.max()+1))
train_label[np.arange(tmp.size),tmp] = 1


fileName = sys.argv[2]

data = np.loadtxt(open(fileName, 'rb'), delimiter=',')
test_data = data[:, :784]

tmp = data[:, 784].astype(int)
test_label = np.zeros((tmp.size, tmp.max()+1))
test_label[np.arange(tmp.size),tmp] = 1


'''Train the network for the number of iterations'''
'''Implement function to measure the accuracy'''
batch = 64
size = train_data.shape[0]

train_losses = []


for _ in range(5000):
    for j in range(0, size, batch):
        Network.Train(train_data[j:j+batch], train_label[j:j+batch])    
    #print("Loss: ", Network.loss/size)
    #print("Iteration: ", _, Network.Accuracy(test_data, test_label))
    
    train_losses.append(Network.loss/size)
    Network.loss = 0

print(Network.Accuracy(train_data, train_label))
print(Network.Accuracy(test_data, test_label))
print(5000)
print(learning_rate)