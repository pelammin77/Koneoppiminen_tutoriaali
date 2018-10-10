"""
file: tf_neural_network.py
author: Petri Lamminaho
Simple neural network with TensorFlow.
Network has one hidden layer
"""
import tensorflow as tf
import numpy as np

print(tf.__version__) #prints TensorFlow's version
#imputs
x_data = np.array([
[0.0,0.0], [0.0,1.0], [1.0,0.0], [1.0,1.0]
])
# outputs
y_data = np.array([
[0.0], [1.0], [1.0], [0.0]
])

lr = 0.1 # leaning rate
num_inputs = 2 # dim of inputs
num_hidden_layer_neurons = 4 # num of neurons in hidden layer
num_outputs = 1
epocs = 10000 # num of training epochs

X = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
w0 = tf.Variable(tf.random_uniform([num_inputs, num_hidden_layer_neurons], -1.0, 1.0)) # init weights
w1 = tf.Variable(tf.random_uniform([num_hidden_layer_neurons, num_outputs], -1.0, 1.0)) # init weights
b1 = tf.Variable(tf.ones([num_hidden_layer_neurons])) # first  bias
b2 = tf.Variable(tf.ones([num_outputs]))  #second bias


hidden_layer = tf.sigmoid(tf.matmul(X, w0)+ b1) # activate hidden layer
output_layer = tf.sigmoid(tf.matmul(hidden_layer, w1) + b2) # activate output
cost = tf.reduce_mean(-y*tf.log(output_layer) - (1-y)*tf.log(1 - output_layer) ) # loss
optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for step in range(epocs):
        sess.run(optimizer, feed_dict={X:x_data, y: y_data })
        if step % 1000 == 0:
            print("Cost:",sess.run(cost, feed_dict={X:x_data, y: y_data }))
    res = tf.equal(tf.floor(output_layer + 0.5), y)
    acc = tf.reduce_mean(tf.cast(res, "float")*100)
    print(sess.run([output_layer], feed_dict={X:x_data, y: y_data } ))
    print("Accuracy: ", acc.eval({X: x_data, y: y_data}),"%")
    