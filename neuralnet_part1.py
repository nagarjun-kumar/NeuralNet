
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

        For Part 1 the network should have the following architecture (in terms of hidden units):

        in_size -> 32 ->  out_size
        We recommend setting the lrate to 0.01 for part 1

        """
        super(NeuralNet, self).__init__()
        self.lrate = lrate
        self.loss_fn = loss_fn
        self.in_size = in_size
        self.out_size = out_size
        self.model = nn.Sequential(nn.Linear(in_size, 512), nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024,1024), nn.Linear(1024, out_size))
        self.optimizer = optim.Adam(self.model.parameters(), lr = lrate)


    def forward(self, x):
        """ A forward pass of your neural net (evaluates f(x)).

        @param x: an (N, in_size) torch tensor

        @return y: an (N, out_size) torch tensor of output from the network
        """
        return self.model(x)

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
        return float(loss.data)


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
    """
    neural = NeuralNet(0.0001, nn.CrossEntropyLoss(),train_set.shape[1], 2)
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


