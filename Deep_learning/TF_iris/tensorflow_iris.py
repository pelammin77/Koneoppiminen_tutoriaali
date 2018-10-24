"""
file: tensorflow_iris.py
author: Petri Lamminaho
Neural network that classifiers Iris data
Uses TensorFlow
"""

import numpy as np
import tensorflow as tf
import pandas as pd


def label_to_vect(label):
    vect = []
    if label == "Iris-setosa":
        vect = [1, 0, 0]
    elif label == "Iris-versicolor":
        vect = [0, 1, 0]
    elif label == "Iris-virginica":
        vect = [0, 0, 1]
    return vect


def pre_processing_data(file):
    X = []
    y = []
    data_file = open(file, 'r')
    for line in data_file.read().strip().split('\n'):
        line = line.split(',')
        X.append([line[0], line[1], line[2], line[3]])
        y.append(label_to_vect(line[4]))
    return X, y


"""
# data from Pandas 

def  preprocessing_data(data):
    data = pd.read_csv(data, names=['f1','f2','f3','f4','label'])
    s = np.asarray([1, 0, 0])
    ve = np.asarray([0, 1, 0])
    vi = np.asarray([0, 0, 1])
    data['label'] = data['label'].map({'Iris-setosa': s, 'Iris-versicolor': ve, 'Iris-virginica': vi})
    return data
"""

train_X, train_y = pre_processing_data('iris.train')
test_x, test_y = pre_processing_data("iris.test")


# create neural network
def neural_net(x, weights, bias):
    h_layer = tf.add(tf.matmul(x, weights['hidden']), bias['hidden'])
    h_layer = tf.nn.relu(h_layer)
    out_layer = tf.matmul(h_layer, weights['output']) + bias['output']
    return out_layer


lr = 0.01
training_epochs = 500
display_epoch = 100
num_inputs = 4
num_outputs = 3
hidden_neurons = 10


X = tf.placeholder(tf.float32, [None, num_inputs])
Y = tf.placeholder(tf.float32, [None, num_outputs])

# defines weights
weights = {
	"hidden" : tf.Variable(tf.random_normal([num_inputs, hidden_neurons]), name="weight_hidden"),
	"output" : tf.Variable(tf.random_normal([hidden_neurons, num_outputs]), name="weight_output")
}

# defines bias
bias = {
	"hidden" : tf.Variable(tf.random_normal([hidden_neurons]), name="bias_hidden"),
	"output" : tf.Variable(tf.random_normal([num_outputs]), name="bias_output")
}

predict = neural_net(X,weights, bias) #forwardbrobagation
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))# gets net loss/error
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)#optimize loss with Adam
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

# training
    for epoch in range(training_epochs):
        _, l = sess.run([optimizer,loss], feed_dict={ X: train_X, Y: train_y}) # run optimizer
        if (epoch+1)%display_epoch == 0:
            print("Epoch: ", (epoch+1), "Loss:", l)
    print("optimization done")

# testing model
    test_result = sess.run(predict, feed_dict={X: test_x})
    correct_result = tf.equal(tf.argmax(test_result, 1), tf.argmax(test_y, 1))
    acc = tf.reduce_mean(tf.cast(correct_result, "float"))
    print("Model acc:", acc.eval({X: test_x, Y: test_y}))
