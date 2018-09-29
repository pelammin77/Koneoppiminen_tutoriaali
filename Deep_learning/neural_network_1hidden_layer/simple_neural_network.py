"""
file: simple_neural_network.py
author: Petri Lammiaho
 Simple neural network with 1 hidden layer
 uses backbrobacation to optimize  weights
"""

import numpy as np


#inputs matrix
x = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]

])

#outputs matrix
y = np.array([
     [0],
     [0],
     [0],
     [1]
])


b1 = 1
b2 = 1
epochs = 100000
step = 1
lr = 0.1

#activation function sigmoid
def sigmoid(x, deriv=False):
    if deriv == True:
        return (x*(1-x))
    return 1/(1+np.exp(-x))

np.random.seed(1)
#creates random weights between -1 - 1
w0 = 2 * np.random.random((2,4)) - b1
w1 = 2 * np.random.random((4,1)) - b2

print("w0:", w0)
print("w1:", w1)

for i in range(epochs):
    input_layer = x
    hidden_layer = sigmoid(np.dot(input_layer, w0))
    output_layer = sigmoid(np.dot(hidden_layer,w1))

    output_error = y - output_layer # calculate error

    if (i%5000)==0:
        print("Err:"+ str(np.mean(np.abs(output_error))))

#backbrobacation
    output_delta = output_error * sigmoid(output_layer, deriv=True)
    hidden_layer_error = output_delta.dot(w1.T)
    hidden_layer_delta = hidden_layer_error * sigmoid(hidden_layer, deriv=True)

#adjust weights
    w1 += hidden_layer.T.dot(output_delta) * lr
    w0 += input_layer.T.dot(hidden_layer_delta)*lr
    step += 1

print("Output:")
print(output_layer)