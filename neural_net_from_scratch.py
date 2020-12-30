# import required libraries
import numpy as np
import pandas as pd
import mnist

# hyper parameters
output_size = 10
num_of_epochs = 100
learning_rate = 0.000005

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
def train_neural_network(x_train, y_train, layers, biases, learning_rate, num_of_epochs):
    for epoch in range(num_of_epochs):
        logit_outs = []
        input = x_train
        for idx, layer in enumerate(layers):
            # forward pass - using numpy dot product
            net_output = np.dot(input, layers[idx]) + biases[idx]
            # activation (non linearity)
            current_logit_out = sigmoid(net_output)
            # save the layer's output for backpropagation
            logit_outs.append(current_logit_out)
            input = current_logit_out

        # calculate error
        errors = current_logit_out - y_train
        loss = errors
        # backwards pass - go through layers in reverse, backpropagate error
        for idx in range(len(layers) - 1, -1, -1):

            backprop_output = sigmoid_derivative(logit_outs[idx])
            backprop_output_prod = loss * backprop_output
            d_loss_bias = np.zeros_like(loss)
            d_loss_bias[np.arange(len(loss)), loss.argmax(1)] = 1
            loss = np.dot(backprop_output_prod, layers[idx].T)
            if idx > 0:
                logit_out_t = logit_outs[idx - 1].T
            else:
                logit_out_t = x_train.T
            layers[idx] = layers[idx] - np.dot(logit_out_t, backprop_output_prod) * learning_rate
            # TODO - update biases
            # for bias_idx, i in enumerate(backprop_output_prod):
            d_loss_bias = np.sum(d_loss_bias, axis=0)
            biases[idx] = biases[idx] - d_loss_bias * learning_rate
        loss = errors.sum()
        print(loss)
    return layers, biases

# def accuracy(x_test, y_test, layers, biases, learning_rate, num_of_epochs):
#     input = x_test
#     for idx, layer in enumerate(layers):
#         # forward pass - using numpy dot product
#         net_output = np.dot(input, layers[idx]) + biases[idx]
#         # activation (non linearity)
#         current_logit_out = sigmoid(net_output)
#         input = current_logit_out
#
#     # if np.argmax(current_logit_out) == np.argmax(y_test)

# initialize weights and biases
np.random.seed(42)
layer1 = np.random.rand(784, 60)
layer2 = np.random.rand(60, 10)
bias1 = np.random.rand(60)
bias2 = np.random.rand(10)

# train the model
weight_out, bias_out = train_neural_network(x_train, y_train, [layer1, layer2], [bias1, bias2], learning_rate, num_of_epochs)

# make predictions using model
