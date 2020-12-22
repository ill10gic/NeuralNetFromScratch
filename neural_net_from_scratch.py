# import required libraries
import numpy as np
import pandas as pd
import mnist

# hyper parameters
output_size = 10
num_of_epochs = 1000
learning_rate = 0.00005

# load data; x = features and y = labels/classifications
x_train, y_train = mnist.train_images(), mnist.train_labels()
x_test, y_test = mnist.test_images(), mnist.test_labels()
x_train, x_test = x_train.reshape(-1, 28 * 28), x_test.reshape(-1,
                                                               28 * 28)  # flatten from 28X28 pixel images to a vector of size 784
x_train, x_test = x_train / 255, x_test / 255  # normalize the input values
y_train = np.eye(output_size)[y_train]  # one hot vectors
y_test = np.eye(output_size)[y_test]  # one hot vectors


# define activation function (forward pass) - sigmoid function
def sigmoid(net_input):
    return 1 / (1 + np.exp(-net_input))


# define derivative activation function (backwards pass)
def sigmoid_derivative(neuron_out):
    return sigmoid(neuron_out) * (1.0 - sigmoid(neuron_out))


# define the train function
def train_neural_network(x_train, y_train, weights, bias, learning_rate, num_of_epochs):
    for epoch in range(num_of_epochs):
        # forward pass - using numpy dot product
        net_output = np.dot(x_train, weights) + bias
        # activation (non linearity)
        logit = sigmoid(net_output)
        # calculate error
        errors = logit - y_train
        # backwards pass
        cost_function = errors
        backprop_output = sigmoid_derivative(logit)
        backprop_output_prod = cost_function * backprop_output
        x_train_T = x_train.T
        # update weights
        weights = weights - np.dot(x_train_T, backprop_output_prod) * learning_rate
        loss = errors.sum()
        print(loss)
        for i in backprop_output_prod:
            bias = bias - i * learning_rate
    return weights, bias


# initialize weights and biases
np.random.seed(42)
weights = np.random.rand(784, 10)
bias = np.random.rand(10)

# train the model
weight_out, bias_out = train_neural_network(x_train, y_train, weights, bias, learning_rate, num_of_epochs)

# make predictions using model
