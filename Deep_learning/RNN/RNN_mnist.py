"""
File: RNN_mnist.py
Author Petri Lamminaho
Simple recurrent neural network with TensorFlow and Mnist-dataset
"""


import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

batch_size = 128
num_classes = 10
chunk_size = 28
num_of_chunk = 28
max_epochs = 2
rnn_layer_size = 128

x = tf.placeholder('float', [None, num_of_chunk,chunk_size])
y = tf.placeholder('float')

def rnn_model(x):
    rnn_layer = {
               'Weights': tf.Variable(tf.random_normal([rnn_layer_size, num_classes])),
               'Biases': tf.Variable(tf.random_normal([num_classes]))
               }

    print("x's shape",x.shape)
    x = tf.transpose(x,[1,0,2])
    print("Shape after transpose:",x.shape)
    x = tf.reshape(x,[-1, chunk_size])
    print("Shape after reshape",x.shape)
    x = tf.split(x, num_of_chunk, 0)
    lstm_cell = rnn_cell.BasicLSTMCell(rnn_layer_size, state_is_tuple=True)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    output = tf.matmul(outputs[-1], rnn_layer['Weights']) + rnn_layer['Biases']
    return output


def train(x):
    prediction = rnn_model(x)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epochs in range(max_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape((batch_size, num_of_chunk, chunk_size))
                _, c = sess.run([optimizer, loss], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epochs, 'completed out of', max_epochs, 'loss:', epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

            acc = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:',
                  acc.eval({x: mnist.test.images.reshape((-1, num_of_chunk, chunk_size)), y: mnist.test.labels}))
train(x)