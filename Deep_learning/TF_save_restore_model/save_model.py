"""
file: save_model.py
author: Petri Lamminaho
Simple model dave feed forward neural network to file TensorFlow
 """

import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
model_dir = "model/"
checkpoint_file = "model.ckpt"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = model_dir + checkpoint_file


num_hl1_nodes = 1000
num_hl2_nodes = 1000
num_hl3_nodes = 1000
num_classes= 10
batch_size = 128
x = tf.placeholder('float', [None, 784] )
y = tf.placeholder('float')


def nn_model(data):
    h1_layer = {
        'weights': tf.Variable(tf.random_normal([784,num_hl1_nodes])),
        'biases':tf.Variable(tf.random_normal([num_hl1_nodes]))
    }
    h2_layer = {
        'weights': tf.Variable(tf.random_normal([num_hl1_nodes, num_hl2_nodes])),
        'biases': tf.Variable(tf.random_normal([num_hl2_nodes]))
    }
    h3_layer = {
        'weights': tf.Variable(tf.random_normal([num_hl2_nodes, num_hl3_nodes])),
        'biases': tf.Variable(tf.random_normal([num_hl3_nodes]))
    }
    output_layer = {
        'weights': tf.Variable(tf.random_normal([num_hl3_nodes, num_classes])),
        'biases': tf.Variable(tf.random_normal([num_classes]))
    }

    h_layer_1 = tf.add(tf.matmul(data,h1_layer['weights']), h1_layer['biases'])
    h_layer_1 = tf.nn.relu(h_layer_1)

    h_layer_2 = tf.add(tf.matmul(h_layer_1, h2_layer['weights']), h2_layer['biases'])
    h_layer_2 = tf.nn.relu(h_layer_2)

    h_layer_3 = tf.add(tf.matmul(h_layer_2, h3_layer['weights']), h3_layer['biases'])
    h_layer_3 = tf.nn.relu(h_layer_3)

    output = tf.matmul(h_layer_3, output_layer['weights']) + output_layer['biases']
    return output


max_epochs = 10
predict = nn_model(x)  # forwardbrobagation
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=y))  # gets net loss/error
optimizer = tf.train.AdamOptimizer().minimize(loss)  # optimize loss with Adam
saver = tf.train.Saver()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    try:
        saver.restore(sess, model_path)
        print("Model restored from file: %s" % model_path)
        # Test model
        correct_prediction = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval(
            {x: mnist.test.images, y: mnist.test.labels}))
        print("Continuing training....")

    except:
        print("Can't find  pre trained model ")
        print("Start training")

    for epochs in range(max_epochs):
        epochs_loss = 0
        for _ in range(int(mnist.train.num_examples / batch_size)):
            epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            _, l = sess.run([optimizer, loss], feed_dict={x: epoch_x, y: epoch_y})
            epochs_loss += l
        print('Epoch', epochs, 'completed out of', max_epochs, 'loss:', epochs_loss)

        correct = tf.equal(tf.argmax(predict, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

    save_path = saver.save(sess, model_path)
    print("Model saved to", save_path)
