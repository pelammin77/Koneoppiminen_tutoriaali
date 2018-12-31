"""
File: rnn_exp.py
Author: Petri Lamminaho
Simple RNN-net uses only NumPy
Calculates binary numbers
"""

import copy, numpy as np
np.random.seed(0)

def sigmoid(x):
    output = 1/ (1 + np.exp(-x))
    return output

def sigmoid_derivative(x):
    return x * (1 - x)

int2binary = {}
bits = 8
#binary_string = lambda x: ''.join(reversed([str((x >> i) & 1) for i in range(binary_dim)]))
largest_number = pow(2, bits)
binary = np.unpackbits(
    np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]
# input variables
learning_rate = 0.1
input_nodes = 2
hidden_nodes = 32
output_dim = 1
# initialize weights
w_0 = 2 * np.random.random((input_nodes, hidden_nodes)) - 1
w_1 = 2 * np.random.random((hidden_nodes, output_dim)) - 1
w_hidden = 2 * np.random.random((hidden_nodes, hidden_nodes)) - 1
w0_update = np.zeros_like(w_0)
w1_update = np.zeros_like(w_1)
w_h_update = np.zeros_like(w_hidden)
# training
for j in range(10000):
    a_int = np.random.randint(largest_number / 2)  # random int
    a = int2binary[a_int]  # encode int to binary
    b_int = np.random.randint(largest_number / 2)  # random int
    b = int2binary[b_int]  # encode int to binary
    # true answer
    c_int = a_int + b_int
    c = int2binary[c_int]
    # where we'll store our best guess (binary encoded)
    prediction = np.zeros_like(c)
    overallError = 0
    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_nodes))
    # moving along the positions in the binary encoding
    for position in range(bits):
        # generate input and output
        X = np.array([[a[bits - position - 1], b[bits - position - 1]]])
        y = np.array([[c[bits - position - 1]]]).T
        # hidden layer (input ~+ prev_hidden)
        layer_1 = sigmoid(np.dot(X, w_0) + np.dot(layer_1_values[-1], w_hidden))
        # output layer (new binary representation)
        layer_2 = sigmoid(np.dot(layer_1, w_1))
      # calculate err
        layer_2_error = y - layer_2
        layer_2_deltas.append((layer_2_error) * sigmoid_derivative(layer_2))
        overallError += np.abs(layer_2_error[0])
        # decode estimate so we can print it out
        prediction[bits - position - 1] = np.round(layer_2[0][0])
        # store hidden layer so we can use it in the next timestep
        layer_1_values.append(copy.deepcopy(layer_1))
    future_layer_1_delta = np.zeros(hidden_nodes)
    for position in range(bits):
        X = np.array([[a[position], b[position]]])
        layer_1 = layer_1_values[-position - 1]
        prev_layer_1 = layer_1_values[-position - 2]
        # error at output layer
        layer_2_delta = layer_2_deltas[-position - 1]
        # error at hidden layer
        layer_1_delta = (future_layer_1_delta.dot(w_hidden.T) + layer_2_delta.dot(
            w_1.T)) * sigmoid_derivative(layer_1)
        # let's update all our weights so we can try again
        w1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        w_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        w0_update += X.T.dot(layer_1_delta)
        future_layer_1_delta = layer_1_delta
    w_0 += w0_update * learning_rate
    w_1 += w1_update * learning_rate
    w_hidden += w_h_update * learning_rate
    w0_update = 0
    w1_update = 0
    w_h_update = 0

    # print out progress
    if (j % 1000 == 0):
        print("Error:" + str(overallError))
        print("Pred:" + str(prediction))
        print("True:" + str(c))
        out = 0
        for index, x in enumerate(reversed(prediction)):
            out += x * pow(2, index)
        print(str(a_int) + " + " + str(b_int) + " = " + str(out))
        print("------------")