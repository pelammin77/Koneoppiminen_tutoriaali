"""
file: KerasPolicy_gradient.py
Author: Petri Lamminaho
Desc: Policy gradient agent Keras
"""

from keras.layers import Dense, Activation, Input
from keras.models import Model, load_model
from keras.optimizers import Adam
import keras.backend as K
import numpy as np

class Agent(object):
    def __init__(self, ALPHA, GAMMA=0.99, n_actions=2,
                 layer1_size=16, layer2_size=16, input_dims=4,
                 fname='reinforce.h5'):
        self.gamma = GAMMA
        self.lr = ALPHA
        self.G = 0
        self.input_dims = input_dims
        self.fc1_size = layer1_size
        self.fcl2_size = layer2_size
        self.n_actions = n_actions
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.policy_net, self.predict = self.build_policy_network()
        self.policy_net.summary()
        self.predict.summary()
        self.action_space = [i for i in range(n_actions)]

        self.model_file = fname

    def build_policy_network(self):
        input = Input(shape=(self.input_dims,))
        updates = Input(shape=[1])
        dense1 = Dense(self.fc1_size, activation='relu')(input)
        dense2 = Dense(self.fcl2_size, activation='relu')(dense1)
        output = Dense(self.n_actions, activation='softmax')(dense2)

        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            log_likehood = y_true * K.log(out)


            return K.sum(-log_likehood * updates)

        policy_net = Model(input=[input, updates], output=[output])
        policy_net.compile(optimizer=Adam(lr=self.lr), loss=custom_loss)
        predict = Model(input=[input], output=[output])
        return policy_net, predict

    def choose_action(self, observation):
        state = observation[np.newaxis, :]
        probabilities = self.predict.predict(state)[0]
        action = np.random.choice(self.action_space, p=probabilities)
        return action

    def store_memory(self, observation, action, reward):
        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    def learn(self):
        state_memory = np.array(self.state_memory)
        action_memory = np.array(self.action_memory)
        reward_memory = np.array(self.reward_memory)

        actions = np.zeros([len(action_memory), self.n_actions])# [0,0
        actions[np.arange(len(action_memory)), action_memory] = 1 #[1,0] [0,1
        G = np.zeros_like(reward_memory)
        for t in range(len(reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(reward_memory)):
                G_sum += reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        self.G = (G - mean) / std

        loss = self.policy_net.train_on_batch([state_memory, self.G], actions)

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

        return loss

    def save_model(self):
        self.policy_net.save(self.model_file)

    def load_model(self):
        self.policy_net = load_model(self.model_file)

import gym

if __name__ == '__main__':

    agent = Agent(ALPHA=0.0005, input_dims=4, GAMMA=0.99,
                      n_actions=2, layer1_size=16, layer2_size=16)

    env = gym.make('CartPole-v0')
    score_history = []

    num_episodes = 2000

    for i in range(num_episodes):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.store_memory(observation, action, reward)
            observation = observation_
            score += reward
        score_history.append(score)

        loss = agent.learn()

        print('episode: ', i,'score: %.1f' % score,
                'average score %.1f' % np.mean(score_history[max(0, i - 100):(i + 1)]))

