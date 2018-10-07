"""
File: 1_layer_neural_network.py
Author Petri Lamminaho
Simplest neural network
Only input and output. None hidden layer
"""
import numpy as np

class Net():
#--------------------------------
    def __init__(self):
        """
        constructor
        """
        np.random.seed(1)
        self. __b1= 1
        self. __weights = 2* np.random.random((3, 1)) - self.__b1
#________________________________________________________________________
    def print_weights(self):
        """
        Prints weights
        """
        print(self.__weights)
#___________________________________________________________________
    def __sigmoid(self, x, deriv=False):
        """
        :param x:
        :param deriv:
        :return:
        Activation function
        """
        if deriv == True:
            return (x * (1 - x))
        return 1 / (1 + np.exp(-x))
#----------------------------------------------------------------
    def train(self, training_inputs, training_outputs, epochs):
        """
        :param training_inputs:
        :param training_outputs:
        :param epochs:
        Train the network
        adjusting weights
        """
        for iter in range(epochs):
            outputs = self.pred(training_inputs)
            loss = training_outputs - outputs
            delta = np.dot(training_inputs.T, loss * self.__sigmoid(outputs, deriv=True))
            self.__weights += delta

#--------------------------------------------------------------------------
    def pred(self, inputs):
        """

        :param inputs:
        :return: prediction
        Makes prediction uses adjusted weights
        """
        return self.__sigmoid(np.dot(inputs, self.__weights))
#---------------------------------------------------------------
if __name__ == "__main__":
    nn = Net() # creates object
    print("Starting weights are:")
    nn.print_weights()
    training_set_inputs = np.array([[0, 0, 0], [1, 1,1], [1, 0, 1], [0, 1,0]])
    training_set_outputs = np.array([[0, 1, 1, 0]]).T
    nn.train(training_set_inputs, training_set_outputs, 100000)
    print("Weights after training:")
    nn.print_weights()
    print("Output:")
    new_data = np.array([1,0,0])
    print(nn.pred(new_data))


