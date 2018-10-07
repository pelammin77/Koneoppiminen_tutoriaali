"""
file: 3_layer_net.py
Author: Petri Lamminaho
email: lammpe77@gmail.com
Neural network with two hidden layers
"""
from numpy import random, array, exp, dot


class Triple_layer_neural_network():

    def __init__(self):
        """
        constructor
        """
        random.seed(1)
        num_peers_hidden_layer_1 = 5
        num_peers_hidden_layer_2 = 4
        b1 = 1
        b2 = 1
        b3  = 1

        self.weights1 = 2 * random.random((3, num_peers_hidden_layer_1)) - b1
        self.weights2 = 2 * random.random((num_peers_hidden_layer_1, num_peers_hidden_layer_2)) - b2
        self.weights3 = 2 * random.random((num_peers_hidden_layer_2, 1)) - b3
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
    def train(self, training_inputs, training_outputs, num_of_training_iterations):
        for iteration in range(num_of_training_iterations):

            input_layer = training_inputs
            hidden_layer_1_activation = self.__sigmoid(dot(input_layer, self.weights1))
            hidden_layer_2_activation = self.__sigmoid(dot(hidden_layer_1_activation, self.weights2))
            output = self.__sigmoid(dot(hidden_layer_2_activation, self.weights3))

            # calculate 'error'
            err4 = (training_outputs - output) * self.__sigmoid_derivative(output)
            err3 = dot(self.weights3, err4.T) * (self.__sigmoid_derivative(hidden_layer_2_activation).T)
            err2 = dot(self.weights2, err3) * (self.__sigmoid_derivative(hidden_layer_1_activation).T)

            # get adjustments (gradients) for each layer
            delta3 = dot(hidden_layer_2_activation.T, err4)
            delta2 = dot(hidden_layer_1_activation.T, err3.T)
            delta1 = dot(training_inputs.T, err2.T)

            # adjust weights accordingly
            self.weights1 += delta1
            self.weights2 += delta2
            self.weights3 += delta3
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
        hidden_layer_1_activation = self.__sigmoid(dot(input_layer, self.weights1))
        hidden_layer_2_activation = self.__sigmoid(dot(hidden_layer_1_activation, self.weights2))
        output = self.__sigmoid(dot(hidden_layer_2_activation, self.weights3))
        return output
#-------------------------------------------------------------------------------------------------------
"""
main function 
"""
if __name__ == "__main__":
    nn = Triple_layer_neural_network()
    print("random start weights")
    print(nn.weights1)
    training_data_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1],[0,1,0]])
    training_data_outputs = array([[0, 1, 1, 0, 0]]).T
    print("Training data inputs:")
    print(training_data_inputs)
    print("Training data outputs:")
    print(training_data_outputs)
    nn.train(training_data_inputs, training_data_outputs, 10000)
    print("New weights after training: ")
    print(nn.weights1)
    # Test the neural network with a new input
    print ("Trying new input data [1, 0, 0 ] -> ?: ( output should be close 1")
    print("result:", nn.pred(array([1, 0, 0]))) #output: 0.99650838
