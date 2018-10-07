"""
file: 2_layer_net.py
Author: Petri Lamminaho
email: lammpe77@gmail.com
Neural network with one hidden layer
"""
from numpy import random, array, exp, dot
import numpy as np


class Double_layer_neural_network():

    def __init__(self):
        """
        constructor
        """
        random.seed(1)
        num_peers_hidden_layer = 4
        b1 = 1
        b2 = 1

        self.weights1 = 2 * random.random((3, num_peers_hidden_layer)) - b1
        self.weights2 = 2 * random.random((num_peers_hidden_layer, 1)) - b2

#-----------------------------------------------------------------------------------------------------------
    def __sigmoid(self, x):
        """
        private function
        sigmoid function pass the data
         and normalize data to 1 or 0
        :param x:
        :return: 1 or 0
        """
        return 1 / (1 + exp(-x))
#--------------------------------------------------------------------------------------------------
    def __sigmoid_derivative(self, x):
        """
        private function
        :param x:
        :return derivative function :
        """
        return x * (1 - x)
#----------------------------------------------------------------------------------
    def train(self, training_inputs, training_outputs, epochs):
        """

        :param training_inputs:
        :param training_outputs:
        :param epochs:
        Training
        Adjust weights

        """
        for i in range(epochs):
            input_layer = training_inputs
            hidden_layer = self.__sigmoid(dot(input_layer, self.weights1))
            output_layer = self.__sigmoid(dot(hidden_layer, self.weights2))
            output_error = training_outputs - output_layer  # calculate error
            if (i % 1000) == 0:
                print("Err:" + str(np.mean(abs(output_error))))


            # backbrobacation
            output_delta = output_error * self.__sigmoid_derivative(output_layer)
            hidden_layer_error = output_delta.dot(self.weights2.T)
            hidden_layer_delta = hidden_layer_error * self.__sigmoid_derivative(hidden_layer)

            # adjust weights
            self.weights2 += hidden_layer.T.dot(output_delta)
            self.weights1 += input_layer.T.dot(hidden_layer_delta)

  #---------------------------------------------------------------------------------------------
    def pred(self, inputs):
        """

        :param inputs:
        :return: output
        take outputs
        pass inputs to next layer
        returns output
         """
        input_layer = inputs
        hidden_layer_activation = self.__sigmoid(dot(input_layer, self.weights1))
        output = self.__sigmoid(dot(hidden_layer_activation, self.weights2))
        return output


#-------------------------------------------------------------------------------------------------------
"""
main function 
"""

if __name__ == "__main__":
    nn = Double_layer_neural_network()
    print("random start weights")
    print(nn.weights2)
    training_data_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_data_outputs = array([[0, 1, 1, 0]]).T
    nn.train(training_data_inputs, training_data_outputs, 10000)
    print("New weights after training: ")
    print(nn.weights1)
    print(nn.weights2)
    print ("Trying new input data [1, 0, 0 ] -> ?: ( output should be close 1")
    print("result:", nn.pred(array([1, 0, 0]))) #output: 0.99742832
