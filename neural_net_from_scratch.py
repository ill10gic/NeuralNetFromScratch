# import required libraries
import numpy as np
import pandas as pd
import mnist

# hyper parameters
output_size = 10
num_of_epochs = 20
batch_size = 32
learning_rate = 0.005

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
    num_batches_per_epoch = len(x_train) // batch_size
    losses, accuracies = [], []
    for epoch in range(num_of_epochs):
        epoch_loss = 0
        shuffler = np.random.permutation(len(x_train))
        x_train = x_train[shuffler]
        y_train = y_train[shuffler]
        for b in range(num_batches_per_epoch):
            # Get batch:
            batch_index_begin = b * batch_size
            batch_index_end = batch_index_begin + batch_size
            x = x_train[batch_index_begin: batch_index_end]
            targets = y_train[batch_index_begin: batch_index_end]
            logit_outs = []
            input = x
            for idx, layer in enumerate(layers):
                # forward pass - using numpy dot product
                net_output = np.dot(input, layers[idx]) + biases[idx]
                # activation (non linearity)
                current_logit_out = sigmoid(net_output)
                # save the layer's output for backpropagation
                logit_outs.append(current_logit_out)
                input = current_logit_out
            loss = current_logit_out - targets
            epoch_loss += loss

            # backwards pass - go through layers in reverse, backpropagate error
            for idx in range(len(layers) - 1, -1, -1):
                backprop_output = sigmoid_derivative(logit_outs[idx])
                backprop_output_prod = loss * backprop_output
                loss = np.dot(backprop_output_prod, layers[idx].T)
                if idx > 0:
                    logit_out_t = logit_outs[idx - 1].T
                else:
                    logit_out_t = x.T
                layers[idx] = layers[idx] - np.dot(logit_out_t, backprop_output_prod) * learning_rate
                for i in backprop_output_prod:
                    biases[idx] = biases[idx] - i * learning_rate
        print(epoch_loss.sum())
        accuracy(x_test, y_test, layers, biases)
    return layers, biases

def accuracy(x_test, y_test, layers, biases):
    input = x_test
    for idx, layer in enumerate(layers):
        # forward pass - using numpy dot product
        net_output = np.dot(input, layers[idx]) + biases[idx]
        # activation (non linearity)
        current_logit_out = sigmoid(net_output)
        input = current_logit_out



    num_correct = 0
    for idx, output in enumerate(current_logit_out):
        if (output.argmax() == y_test[idx].argmax()):
            num_correct += 1

    print('num_correct: {}'.format(num_correct))
    acc = num_correct / len(y_test)
    print('Accuracy: {acc:.2%}'.format(acc=acc))
    return acc


# initialize weights and biases
#np.random.seed(42)
layer1 = np.random.rand(784, 100)
layer2 = np.random.rand(100, 10)
bias1 = np.random.rand(100)
bias2 = np.random.rand(10)

# train the model
weight_out, bias_out = train_neural_network(x_train, y_train, [layer1, layer2], [bias1, bias2], learning_rate, num_of_epochs)

# make predictions using model
print('Final Accuracy')
acc = accuracy(x_test, y_test, weight_out, bias_out)

