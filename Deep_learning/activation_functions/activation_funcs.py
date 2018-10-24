"""
file: activation_funcs.py
author: Petri Lamminaho
Deep learning activation functions
uses only Numpy
"""
import numpy as np

def step(x):
    return 1 * (x > 0)

def step_derivative(x):
    if x != 0:
        return 0



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative( x):
    return x * (1 - x)



def tanh(x):
    return ( 2 / (1 + np.exp(-2*x)) - 1)

    # print( ((1- np.exp(-2 * x)) / (1 + np.exp(-2 * x)))) # this also gives tanh
    # print( (np.tanh(x))) # Numpy's tanh

def tanh_derivative(x):
    return 1. - x * x


def relu(x):
    return x * (x > 0)

def derivative_relu(x):
    return 1. * (x > 0)



def leaky_relu(x, epsilon=0.1):
    if x< 0:
        return epsilon * x
    return x

def derivative_leaky_relu(x, epsilon=0.1):
    if x<0:
        return epsilon
    return 1



def softmax(logits):
    exps = [np.exp(i) for i in logits]
    sum_of_exps = np.sum(exps)
    softmax = [j / sum_of_exps for j in exps]
    return exps, softmax

### test

logits = [4.0, 2.0, 0.2]
#ex, sm = softmax(logits)

#print(step(-0.1)) #0
#print(step(0.1))  #1
# print(ex)
# print(sm)
tanh(5)

#print(step_derivative(5))
