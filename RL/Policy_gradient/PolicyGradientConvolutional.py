"""
file: PolicyGradientConvolutional.py
Author: Petri Lamminaho
Desc: CNN Policy gradient agent Tensorflow
"""


import os
import tensorflow as tf
import numpy as np
import gym


class PolicyGradientConvAgent():
    def __init__(self, ALPHA=0.001, GAMMA=0.95, n_actions=2,
                 fcl_size = 256, input_shape=(100,100), channels=1,
                 chkpt_dir='tmp/checkpoints'):
        self.lr = ALPHA
        self.gamma = GAMMA
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]
        self.fcl_size = fcl_size
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.input_width = input_shape[0]
        self.input_heigth = input_shape[1]
        self.channels = channels
        self.sess = tf.Session()
        self.build_net()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(chkpt_dir, 'policy_network.ckpt')

    def build_net(self):
        with tf.variable_scope('parameters'):
            self.input = tf.placeholder(tf.float32,
                                        shape=[None, self.input_heigth, self.input_width,
                                               self.channels],
                                        name='input')
            self.label = tf.placeholder(tf.int32,
                                        shape=[None], name='label')
            self.G = tf.placeholder(tf.float32, shape=[None, ], name='G')

        with tf.variable_scope('conv_layer_1'):
            conv1 = tf.layers.conv2d(inputs=self.input, filters=32,
                                     kernel_size=(8, 8), strides=4, name='conv1',
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
            batch1 = tf.layers.batch_normalization(inputs=conv1, epsilon=1e-5,
                                                   name='batch1')
            conv1_activated = tf.nn.relu(batch1)

        with tf.variable_scope('conv_layer_2'):
            conv2 = tf.layers.conv2d(inputs=conv1_activated, filters=64,
                                     kernel_size=(4, 4), strides=2, name='conv2',
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
            batch2 = tf.layers.batch_normalization(inputs=conv2, epsilon=1e-5,
                                                   name='batch2')
            conv2_activated = tf.nn.relu(batch2)

        with tf.variable_scope('conv_layer_3'):
            conv3 = tf.layers.conv2d(inputs=conv2_activated, filters=128,
                                     kernel_size=(3, 3), strides=1, name='conv3',
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
            batch3 = tf.layers.batch_normalization(inputs=conv3, epsilon=1e-5)
            conv3_activated = tf.nn.relu(batch3)

        with tf.variable_scope('fc1'):
            flat = tf.layers.flatten(conv3_activated)
            dense1 = tf.layers.dense(flat, units=self.fcl_size, activation=tf.nn.relu)

        with tf.variable_scope('fc2'):
            dense2 = tf.layers.dense(dense1, units=self.n_actions,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.actions = tf.nn.softmax(dense2, name='actions')

        with tf.variable_scope('loss'):
            negative_log_probability = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=dense2, labels=self.label)
            self.loss = tf.reduce_mean(negative_log_probability * self.G)

        with tf.variable_scope('train'):
            # self.train_op = tf.train.RMSPropOptimizer(self.lr, decay=0.99,
            #                                           momentum=0.0, epsilon=1e-6).minimize(self.loss)
            self.train_op = tf.train.AdamOptimizer(self.lr, epsilon=1e-3).minimize(self.loss)

    def choose_action(self, state):
        state = np.array(state).reshape((-1, self.input_heigth,
                                                     self.input_width,
                                                     self.channels))
        probabilities = self.sess.run(self.actions, feed_dict={self.input: state})[0]
        # whole_probabilities = self.sess.run(self.actions, feed_dict={self.input: state})
        # print(whole_probabilities) # prints  [0.58647335 0.41352665]
        action = np.random.choice(self.action_space, p=probabilities)

        return action

    def store_memory(self, observation, action, reward):
        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    def learn(self):
        state_memory = np.array(self.state_memory).reshape(
                    (-1, self.input_heigth, self.input_width, self.channels))
        action_memory = np.array(self.action_memory)
        reward_memory = np.array(self.reward_memory)

        G = np.zeros_like(reward_memory)
        for time_state in range(len(reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(time_state, len(reward_memory)):
                G_sum += reward_memory[k] * discount
                discount *= self.gamma

            G[time_state] = G_sum
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        G = (G - mean) / std

        _ = self.sess.run(self.train_op,
                          feed_dict={self.input: state_memory,
                                     self.label: action_memory,
                                     self.G: G})
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

    def load_model(self):
        print("...Loading checkpoint...")
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_model(self):

        self.saver.save(self.sess, self.checkpoint_file)


def pre_process(image, w, h):
    import cv2
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, w, h)
    return image

    #return np.mean(screen[15:200, 30:125], axis=2)



def stack_frames(stacked_frames, frame, buffer_size):
    if stacked_frames is None:
        stacked_frames = np.zeros((buffer_size, *frame.shape))
        for idx, _ in enumerate(stacked_frames):
            stacked_frames[idx,:] = frame
    else:
        stacked_frames[0:buffer_size-1,:] = stacked_frames[1:,:]
        stacked_frames[buffer_size-1, :] = frame

    return stacked_frames


if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    agent = PolicyGradientConvAgent(ALPHA=0.0005)
    # agent.load_model()

    score_history = []
    score = 0
    MAX_EPISODES = 3000

    for episode in range(MAX_EPISODES):
        print('episode: ', episode, 'score: ', score)
        done = False

        score = 0
        state = env.reset()
        while not done:
            action = agent.choose_action(state)
            new_state, reward, done, info = env.step(action)
            agent.store_memory(state, action, reward)
            state = new_state
            score += reward

        # score_history.append(score)
        agent.learn()


        # agent.save_model()


