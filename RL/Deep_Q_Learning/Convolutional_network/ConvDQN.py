"""
file: ConvDQN.py
Author: Petri Lamminaho
Desc: Deep Q-Learning agent plays Gym's Mountain Car game
CNN agent Keras library 
"""
import keras
import random
import numpy as np
from collections import deque
import gym
import cv2
import tensorflow as tf
from keras.models import Model
keras.backend.set_image_data_format('channels_first')
from keras import layers
from keras import optimizers
import pickle

STACK_DEPTH = 4
BATCH_SIZE = 32
UPDATE_EVERY = 35
SAVE_MODEL_EVERY = 500
MAX_EPISODES = 1000000
STEPS_PER_EPISODES = 300
SKIP_FRAMES = 4


class MC_Agent:
    def __init__(self, env):
        self.env = env
        self.num_actions = self.env.action_space.n
        self.epsilon = 1
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.000018
        self.gamma = 0.99  # Factor by which future rewards are discounted
        self.memory = deque(maxlen=200000)
        self.learning_rate = 0.00025  # Learning rate for the Neural network
        self.frame_width = 150  # Input image width
        self.frame_height = 100  # Input image height
        self.stack_depth = STACK_DEPTH
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.update_target_weights()
        self.loss = 0


    def create_model(self):
        input_shape = (self.stack_depth, self.frame_height, self.frame_width)
        actions_input = layers.Input((self.num_actions,), name='action_mask')

        frames_input = layers.Input(input_shape, name='input_layer')
        conv_1 = layers.Conv2D(32, (8, 8), strides=4, padding='same'
                               , activation='relu', name='conv_1', kernel_initializer='glorot_uniform',
                               bias_initializer='zeros')(frames_input)

        conv_2 = layers.Conv2D(64, (4, 4), strides=2, padding='same', activation='relu', name='conv_2'
                               , kernel_initializer='glorot_uniform', bias_initializer='zeros')(conv_1)

        conv_3 = layers.Conv2D(64, (3, 3), strides=1, padding='same', name='conv_3', activation='relu'
                               , kernel_initializer='glorot_uniform', bias_initializer='zeros')(conv_2)

        flatten_1 = layers.Flatten()(conv_3)

        dense_1 = layers.Dense(512, activation='relu', name='dense_1',
                               kernel_initializer='glorot_uniform', bias_initializer='zeros')(flatten_1)

        output = layers.Dense(self.num_actions, activation='linear', name='output',
                              kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense_1)
        vectorized_output = layers.Multiply(name='vect_output')([output, actions_input])

        model = Model(input=[frames_input, actions_input], output=[vectorized_output])
        optimizer = optimizers.Adam(lr=self.learning_rate)
        model.compile(optimizer, loss=tf.losses.huber_loss)
        return model

        # input_shape = (self.stack_depth, self.frame_height, self.frame_width)
        #
        # model = Sequential()
        # model.add(layers.Convolution2D(32, 8, 8, subsample=(4, 4), input_shape=input_shape))
        # model.add(Activation('relu'))
        # model.add(layers.Convolution2D(64, 4, 4, subsample=(2, 2)))
        # model.add(Activation('relu'))
        # model.add(layers.Convolution2D(64, 3, 3))
        # model.add(Activation('relu'))
        # model.add(Flatten())
        # model.add(Dense(512))
        # model.add(Activation('relu'))
        # model.add(Dense(self.num_actions))
        # model.compile(loss='mse', optimizer=Adam(lr=0.00001))
        # return model


    def decay_epsilon(self):
        self.epsilon = self.epsilon - self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)


    def update_target_weights(self):

        self.target_model.set_weights(self.model.get_weights())


    def add_memory(self, current_state, action, reward, next_state, done):
        self.memory.append([current_state, action, reward, next_state, done])


    def process_image(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image, (self.frame_width, self.frame_height))
        return image


    def get_best_action(self, cur_state):
        cur_state = np.float32(np.true_divide(cur_state, 255))
        action_mask = np.ones((1, self.num_actions))
        q_values = self.model.predict([cur_state, action_mask])[0]
        best_action = np.argmax(q_values)

        return best_action

    def get_targets(self, batch_size):

        current_states = []
        rewards = []
        actions = []
        new_states = []
        dones = []

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            current_states.append(state)
            rewards.append(reward)
            actions.append(action)
            new_states.append(new_state)
            dones.append(done)


        # convert states to numpy array
        # and normalize data
        current_states = np.array(current_states)
        current_states = np.float32(np.true_divide(current_states, 255))

        new_states = np.array(new_states)
        new_states = np.float32(np.true_divide(new_states, 255))

        action_mask = np.ones((1, self.num_actions))
        action_mask = np.repeat(action_mask, batch_size, axis=0)
        future_Q_values = self.target_model.predict([new_states, action_mask])
        future_Q_values[dones] = 0
        targets = rewards + self.gamma * np.max(future_Q_values, axis=1)
        current_action_one_hot = self.get_one_hot(actions)
        return current_states, current_action_one_hot, targets


    def get_one_hot(self,actions):
        actions = np.array(actions)
        one_hots = np.zeros((len(actions), self.num_actions))

        one_hots[np.arange(len(actions)), actions] = 1

        return one_hots


    def train(self, screen_states, action_mask, targets):
        #self.model.fit([screen_states, action_mask], action_mask * targets[:, None], epochs=1, verbose=0)
        #  y labels = action_mask * targets[:, None]
        # self.loss = self.model.train_on_batch([screen_states, action_mask], labels)
        self.loss = self.model.train_on_batch([screen_states, action_mask], action_mask * targets[:, None])


    def save_model(self,file_name):
        self.model.save(file_name)

    def save_model_weights(self, name):
        self.model.save_weights(name)

#### END OF CLASS############################################

#MAIN
env = gym.make('MountainCar-v0').env
agent = MC_Agent(env)
stack_depth = STACK_DEPTH
render_memory = deque(maxlen=stack_depth)
done = False
training = False
batch_size = BATCH_SIZE

# frames  = 35 * steps
update_weights_every = 35
episodes = MAX_EPISODES
max_steps = STEPS_PER_EPISODES
print(agent.memory.maxlen)

collect_experience = agent.memory.maxlen - 50000
print(collect_experience)
frame_skip = STACK_DEPTH


#List reward for each each episode
ep_reward = []


#Loop for the number of episodes
for episode in range(1,episodes):

    render_memory.clear()
    initial_state = env.reset()
    current_image = env.render(mode = 'rgb_array')
    frame = agent.process_image(current_image)
    frame = frame.reshape(1, frame.shape[0], frame.shape[1])
    current_state = np.repeat(frame, stack_depth, axis=0)
    render_memory.extend(current_state)

    episode_reward = 0
    for step in range(max_steps):

        #Calcualte a new action every 4 frames
        if step % frame_skip == 0:
            if training:
                 agent.epsilon = agent.epsilon - agent.epsilon_decay
                 agent.epsilon = max(agent.epsilon_min, agent.epsilon)

            if np.random.rand() <= agent.epsilon:

                action = env.action_space.sample()
            else:

                action = agent.get_best_action(current_state.reshape(1, current_state.shape[0]
                                   , current_state.shape[1], current_state.shape[2]))

        #Take the new action or repeat the previous one if a new one is not calculated
        new_state, reward, done, _ = env.step(action)

        #Collect, process and store the new frame in the buffer
        next_frame = env.render(mode='rgb_array')
        next_frame = agent.process_image(next_frame)
        render_memory.append(next_frame)


        next_state = np.asarray(render_memory)
        agent.memory.append([current_state, action, reward, next_state, done])

        current_state = next_state

        #If the replay memory has a minimum number of experience samples, start training the network
        if len(agent.memory) == collect_experience:
            training = True
            print('Start training')


        if training:
            states, action_mask, targets = agent.get_targets(batch_size)
            agent.train(states,action_mask, targets)

        episode_reward = episode_reward + reward


        if done:
            break

    #Keeping track of the total episode reward for each episode
    ep_reward.append([episode, episode_reward])

    #Print episode information after each episode
    print("episode: {}/{}, epsilon: {}, episode reward: {}, loss:{}"
      .format(episode, episodes, agent.epsilon, episode_reward, agent.loss))

    #Update target weights each update_threshold episodes
    if training and (episode % update_weights_every) == 0:
        print('Weights updated at epsisode:', episode)
        agent.update_target_weights()

    #Saving the model architecture, it's weights and total episode rewards each save_threshold episodes
    if training and (episode%SAVE_MODEL_EVERY) == 0:
        print('Data saved at epsisode:', episode)
        agent.save_model('./train/models/DQN_CNN_model_{}.h5'.format(episode))
        agent.save_model_weights('./train/weights/DQN_CNN_model_weights_{}.h5'.format(episode))
        pickle.dump(ep_reward, open('./train/rewards/rewards_{}.dump'.format(episode), 'wb'))

env.close()
