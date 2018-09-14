"""
CS 535 Deep Learning Assignment 2
Author: Zhou Fang
Date: 02/18/2018
"""

from __future__ import division
from __future__ import print_function
from matplotlib import pyplot as plt


import sys
import copy
import math
import cPickle
import numpy as np


#=========================================================================================
# This is a class for a LinearTransform layer which takes an input 
# weight matrix W and computes W x as the forward step
#=========================================================================================
class LinearTransform(object):

    def __init__(self, W, b):
        """
        initializing the ReLU
        """
        self.input = 0
        self.W = W # l1[dims, hidden_units],l2[hidden_units, 1]
        self.sumW = 0
        self.b = b # l1[hidden_units, 1], l2[1, 1]
        self.output = 0
        self.dW = 0
        self.db = 0
        self.back = 0
    def forward(self, x):
        """
        DEFINE forward function
        """
        self.input = x # l1[dims, batch_size], l2[hidden_units, batch_size]
        self.output = np.dot(self.W.T, self.input) + self.b
    def backward(
        self, 
        grad_output, 
        learning_rate=0.0, 
        momentum=0.0, 
    ):
        """
        DEFINE backward function
        """   
        self.sumW = np.zeros((self.W.shape[0], self.W.shape[1])) # initial self.sumW
        self.back = np.dot(self.W, grad_output) # compute grad_output: self.back: l1[dims, batch_size], l2[hidden_units, batch_size]   
        for i in range(self.input.shape[1]):
            self.sumW += np.dot(self.input[:,i].reshape(self.W.shape[0], 1), grad_output[:,i].reshape(1, self.W.shape[1]))

        self.dW = momentum * self.dW - learning_rate * self.sumW / self.input.shape[1]
        self.db = momentum * self.db - learning_rate * np.mean(grad_output, axis = 1).reshape(-1, 1)

        self.W += self.dW
        self.b += self.db

#=========================================================================================
# This is a class for a ReLU layer max(x,0)
#=========================================================================================
class ReLU(object):
    def __init__(self):
        """
        initializing the ReLU
        """
        self.input = 0
        self.output = 0
        self.back = 0
    def forward(self, x):
        """
        DEFINE forward function
        """
        self.input = x
        #print(x)
        self.output = np.where(self.input <= 0, 0, self.input)
        #print(self.output.shape)
    def backward(
        self, 
        grad_output, 
        learning_rate=0.0, 
        momentum=0.0, 
    ):
        """
        DEFINE backward function
        """   
        y = copy.deepcopy(self.input)# y is the grad of ReLU function
        y[y > 0] = 1 
        y[y == 0] = 0.5
        y[y < 0] = 0
        self.back = grad_output * y

#=========================================================================================
# This is a class for a sigmoid layer followed by a cross entropy layer, the reason 
# this is put into a single layer is because it has a simple gradient form
#=========================================================================================      
class SigmoidCrossEntropy(object):
    def __init__(self):
        """
        initializing the SigmoidCrossEntropy
        """
        self.input = 0
        self.output = 0
        self.sigmoid = 0
        self.back = 0
        self.loss = 0
        self.label = 0
    def forward(self, x, label):
        """
        DEFINE forward function
        """
        x = x.reshape(-1)
        label = label.reshape(-1)
        self.label = label
        self.input = x	
        self.sigmoid = 1.0/(1 + np.exp(-self.input))
        self.loss = np.where(self.label > 0.5, -np.log(self.sigmoid), -np.log(1 - self.sigmoid))
        self.output = np.where(self.sigmoid <= 0.5, 0, 1)
        self.output = self.output.reshape(-1, 1)
    def backward(
        self, 
        learning_rate=0.0, 
        momentum=0.0, 
    ):
        """
        DEFINE backward function
        """
        self.back = (self.sigmoid - self.label).reshape(1, -1)
        #print(self.back.shape)

#=========================================================================================
# This is a class for one hidden layer neural network including train model, prediction model and evaluation model 
#=========================================================================================              
class MLP(object):

    def __init__(self, W1, b1, W2, b2):
        """
        initializing the network
        """
        self.l1 = LinearTransform(W1, b1)
        self.l2 = LinearTransform(W2, b2)
        self.rl = ReLU()
        self.error = SigmoidCrossEntropy()
        self.predict = 0
        self.loss = 0
    
    # in train function, we train the input training data and backprobagation, get train loss
    def train(
        self,
        x_batch, 
        y_batch, 
        learning_rate, 
        momentum,
    ):
        """
        in train function, there is a forward path followed by a backprobagation path to update W and b
        """
        self.l1.forward(x_batch)
        self.rl.forward(self.l1.output)
        self.l2.forward(self.rl.output)
        self.error.forward(self.l2.output, y_batch)
        self.error.backward(learning_rate, momentum)
        self.l2.backward(self.error.back, learning_rate, momentum)
        self.rl.backward(self.l2.back, learning_rate, momentum)
        self.l1.backward(self.rl.back, learning_rate, momentum)
        self.loss = self.error.loss
    def prediction(
        self,
        x_batch,
        y_batch,
    ):
        """
        in predict function, we predict the label and get the test loss
        """
        self.l1.forward(x_batch)
        self.rl.forward(self.l1.output)
        self.l2.forward(self.rl.output)
        self.error.forward(self.l2.output, y_batch)
        self.predict = self.error.output
        self.loss = self.error.loss



def evaluate(x, y):
    """
    in evaluate function, we test the accuracy of our prediction
    """
    correct = sum(np.equal(x, y))
    accuracy = correct / len(y)
    return float(accuracy)



def runMLP(
    train_x, 
    train_y, 
    test_x, 
    test_y, 
    hidden_units, 
    num_epochs, 
    num_batches,
    learning_rate,
    momentum,
):
    """
    runMLP function can train the data and test the date once.
    """
    test_accuracy_out = []
    num_examples, input_dims = train_x.shape
    batch_size = int(num_examples/num_batches)
    
    # initial W1, b1, W2, b2, normal distribution
    W1 = 1/10 * np.random.randn(input_dims, hidden_units).reshape((input_dims, hidden_units))
    W2 = 1/10 * np.random.randn(hidden_units, 1).reshape((hidden_units, 1))
    b1 = 1/10 * np.random.randn(hidden_units, 1).reshape((hidden_units, 1))
    b2 = 1/10 * np.random.randn(1, 1).reshape((1, 1))
    
    # define the mlp model
    mlp = MLP(W1, b1, W2, b2)
    
    # train and predict the model   
    for epoch in xrange(num_epochs):
        start = 0
        for b in xrange(num_batches):
            total_loss = 0.0
            x_batch = train_x[start : start + batch_size]
            y_batch = train_y[start : start + batch_size]
            start += batch_size
            mlp.train(x_batch.T, y_batch, learning_rate, momentum)
            total_loss = np.sum(mlp.loss)
            print(
                '\r[Epoch {}, numbers of batch {}, Hidden units {}, Learning rate {}]    Avg.Loss = {:.3f}'.format(
                    epoch + 1,
                    b + 1,
                    hidden_units,
                    learning_rate,
                    total_loss/batch_size,
                ),
                end='',
            )
            sys.stdout.flush()
        print()
        mlp.prediction(train_x.T, train_y)
        train_loss = np.sum(mlp.loss)/num_examples
        train_accuracy = evaluate(mlp.predict, train_y)
        print('    Train Loss: {:.3f}    Train Acc.: {:.2f}%'.format(
            train_loss,
            100. * train_accuracy,
        ))
        mlp.prediction(test_x.T, test_y)
        test_loss = np.sum(mlp.loss)/test_y.shape[0]
        test_accuracy = evaluate(mlp.predict, test_y)
        print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(
            test_loss,
            100. * test_accuracy,
        ))
        test_accuracy_out.append(test_accuracy)
    return test_accuracy_out



def drawPlot(num_epochs, test_acc_out, parameter_list, parameterName):
    """
    in drawPlot function, we plot the test accuracy with different parameters and save it.
    """
    color_list = ['-go', '-rs', '-bp', '-cd', '-k*', '-yv']
    plt.figure()
    for i, parameter in enumerate(parameter_list):
        print()
        plt.plot(range(1,num_epochs + 1), test_acc_out[i], color_list[i], label = "%s %s" %(parameterName, str(parameter)))
        plt.xlim(0, num_epochs)
        plt.ylim(0.45, 0.85)
    
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.grid(True)
    plt.legend(loc = 'lower right')
    plt.title("Test Accuracy With Different %s" %(parameterName))
    plt.savefig("test_accuracy_%s.jpg" %(parameterName))



if __name__ == '__main__':

    # data load
    data = cPickle.load(open('cifar_2class_py2.p', 'rb'))
    # data normalization
    train_mean = np.mean(data['train_data'], axis = 0).reshape(1, -1)
    train_std =np.std(data['train_data'], axis = 0).reshape(1, -1)
    train_x = (data['train_data'] - train_mean) / train_std
    train_y = data['train_labels']
    test_x =  (data['test_data'] - train_mean) / train_std
    test_y = data['test_labels']
   

    #########################################
    # Run code once with fix parameters (problem 3 and 4)
    #########################################

    # parameter define
    hidden_units = 100
    num_epochs = 20
    num_batches = 1000
    learning_rate = 0.001
    momentum = 0.8

    # run code
    test_acc = runMLP(train_x, train_y, test_x, test_y, hidden_units, num_epochs, num_batches, learning_rate, momentum)



    #########################################
    # Run code once with different numbers of batch size (problem 5)
    #########################################

    # parameter define
    hidden_units = 100
    num_epochs = 20
    num_batches_list = [10, 20, 100, 200, 1000, 2000]
    learning_rate = 0.001
    momentum = 0.8
    test_acc_out = []
    batch_size = []

    # run code
    for num_batches in num_batches_list:
        batch_size.append(int(train_x.shape[0]/num_batches))
        test_acc = runMLP(train_x, train_y, test_x, test_y, hidden_units, num_epochs, num_batches, learning_rate, momentum)
        test_acc_out.append(test_acc)

    # plot
    drawPlot(num_epochs, test_acc_out, batch_size, "Batch Size")



    #########################################
    # Run code once with different learning rate (problem 5)
    #########################################

    # parameter define
    hidden_units = 100
    num_epochs = 20
    num_batches = 100
    learning_rate_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
    momentum = 0.8
    test_acc_out = []

    # run code
    for learning_rate in learning_rate_list:
        test_acc = runMLP(train_x, train_y, test_x, test_y, hidden_units, num_epochs, num_batches, learning_rate, momentum)
        test_acc_out.append(test_acc)

    # plot
    drawPlot(num_epochs, test_acc_out, learning_rate_list, "Learning Rate")

   


    #########################################
    # Run code once with different numbers of hidden units (problem 5)
    #########################################

    # parameter define
    hidden_units_list = [1, 10, 50, 100, 200, 500]
    num_epochs = 20
    num_batches = 100
    learning_rate = 0.001
    momentum = 0.8
    test_acc_out = []

    # run code
    for hidden_units in hidden_units_list:
        test_acc = runMLP(train_x, train_y, test_x, test_y, hidden_units, num_epochs, num_batches, learning_rate, momentum)
        test_acc_out.append(test_acc)

    # plot
    drawPlot(num_epochs, test_acc_out, hidden_units_list, "Hidden Units")
   

















    
