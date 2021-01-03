# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
"""
This is the main entry point for MP6. You should only modify code
within this file and neuralnet_part1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim


class NeuralNet(torch.nn.Module):
    def __init__(self, lrate,loss_fn,in_size,out_size):
        """
        Initialize the layers of your neural network

        @param lrate: The learning rate for the model.
        @param loss_fn: A loss function defined in the following way:
            @param yhat - an (N,out_size) tensor
            @param y - an (N,) tensor
            @return l(x,y) an () tensor that is the mean loss
        @param in_size: Dimension of input
        @param out_size: Dimension of output
        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.c1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.c2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*25, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,out_size)

        self.optimizer = optim.Adam(self.parameters(), self.lrate, weight_decay= 0.001)


    def forward(self, x):
        """ A forward pass of your neural net (evaluates f(x)).

        @param x: an (N, in_size) torch tensor

        @return y: an (N, out_size) torch tensor of output from the network
        """
        x = x.view(len(x),3,32,32)
        x = self.pool(func.relu(self.c1(x)))
        x = self.pool(func.relu(self.c2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @param y: an (N,) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """
        self.optimizer.zero_grad()
        loss = self.loss_fn(self.forward(x),y)
        loss.backward()
        self.optimizer.step()
        return loss.detach().item()


def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) torch tensor
    @param train_labels: an (N,) torch tensor
    @param dev_set: an (M,) torch tensor
    @param n_iter: int, the number of iterations of training
    @param batch_size: The size of each batch to train on. (default 100)

    # return all of these:

    @return losses: Array of total loss at the beginning and after each iteration. Ensure len(losses) == n_iter
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: A NeuralNet object

    # NOTE: This must work for arbitrary M and N

    model's performance could be sensitive to the choice of learning_rate. We recommend trying different values in case
    your first choice does not seem to work well.
    """
    neural = NeuralNet(0.01, nn.CrossEntropyLoss(),train_set.shape[1], 2)
    loss = []
    num_batch = train_set.size()[0]//batch_size
    for i in range(len(train_set)):
        std = train_set[i].std()
        mean_val = train_set[i].mean()
        train_set[i] = (train_set[i] - mean_val)/std
    for i in range(len(dev_set)):
        std = dev_set[i].std()
        mean_val = dev_set[i].mean()
        dev_set[i] = (dev_set[i] - mean_val)/std

    for epoch in range(n_iter):
        start = (epoch % num_batch) * batch_size
        x_val = train_set[start:start + batch_size]
        y_val = train_labels[start:start+batch_size]
        val = neural.step(x_val, y_val)
        loss.append(val)

    yhat = torch.max(neural(dev_set), 1)[1]
    yhat = yhat.tolist()
    return loss,yhat, neural
