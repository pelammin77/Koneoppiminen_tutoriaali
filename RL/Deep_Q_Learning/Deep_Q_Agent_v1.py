"""
file: Deep_Q_Agent_v1.py
Author: Petri Lamminaho
Desc: Deep Q-Learning agent trains andplays Gym's Mountain Car game
Uses Keras Feed forward net not a Convolutional net 
"""

import gym
import numpy as np
import random
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from collections import deque

BATCH_SIZE = 32
NUM_EPOCHS = 1
NUM_TRIALS = 5000


class DQ_Net:
    def __init__(self, env,  load_model_fl =""):
        self.env = env
        self.replay_mem = deque(maxlen=5000)
        self.epsilon = 1
        self.min_epsilon = 0.01
        self.gamma = 0.99
        self.epsilon_decay = 0.0001
        self.lr = 0.01
        self.tau = 0.125

        self.model = self.create_model()
        self.model.load_weights("success.h5")
        self.target_model = self.create_model()


    def create_model(self):

        model = Sequential()
        input_shape = self.env.observation_space.shape
        model.add(Dense(24, input_dim=input_shape[0], activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mse", optimizer=Adam(lr=self.lr))

        return model

    def choose_action(self,state):
        self.epsilon -= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.min_epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])

    def add_replay_mem(self, state, action, reward, new_state, done):
        self.replay_mem.append([state, action, reward, new_state, done])

    def replay(self):
        if len(self.replay_mem) < BATCH_SIZE:
            return
        training_samples = random.sample(self.replay_mem, BATCH_SIZE)

        for sample in training_samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward

            else:
                future_Q = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + future_Q * self.gamma
            #self.model.fit(state, target, epochs=NUM_EPOCHS, verbose=0)
            self.model.train_on_batch(state,target)




    def set_target_weights(self):
        model_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = model_weights[i] * self.tau + target_weights[i] * (1 - self.tau)

        self.target_model.set_weights(target_weights)

    def load_weights(self, file_name):
        self.load_weights(file_name)

    def load_model(self, file_name):
        self.load_model(file_name)


    def save_model(self, file_name):
        self.model.save(file_name)

    def save_weights(self, file_name):
        self.model.save_weights(file_name)


def continuous_MC():
    env = gym.make("MountainCar-v0")
    state_size = env.observation_space.shape[0]

    episodes = NUM_TRIALS
    trial_len = 500
    dqn_agent = DQ_Net(env=env)


    for episode in range(episodes):
        cur_state = env.reset().reshape(1, state_size)
        for step in range(trial_len):
            if episode % 100 == 0:
                env.render()
            action = dqn_agent.choose_action(cur_state)
            new_state, reward, done, _ = env.step(action)

            new_state = new_state.reshape(1, state_size)
            dqn_agent.add_replay_mem(cur_state, action, reward, new_state, done)

            dqn_agent.replay()  # internally iterates default (prediction) model
            dqn_agent.set_target_weights()  # iterates target model

            cur_state = new_state
            if done:
                break
        if step >= 199:
            print("Failed in {} trials, Epsilon is: {}".format(episode, dqn_agent.epsilon))
            if episode % 10 == 0:
                print("Saving model")
                dqn_agent.save_weights("trial_weigts-{}.h5".format(episode))
        else:
            print("---------------")
            print("Completed in {} trials,Epsilon is: {}".format(episode, dqn_agent.epsilon))
            print("Step needed", step)
            print("Saving model")
            print("---------------")
            dqn_agent.save_model("success-{}.h5".format(episode))
            dqn_agent.save_weights('success_weights-{}.h5'.format(episode))

def get_discrete_state(state, env, disc_obs_chunck_size ):
    discrete_state = (state - env.observation_space.low) / disc_obs_chunck_size
    return discrete_state.astype(np.int)



def disc_MC():
    env = gym.make('MountainCar-v0')
    DISC_OBSERVATION_SIZE = [20] * len(env.observation_space.low)
    disc_obs_chunck_size = (env.observation_space.high - env.observation_space.low) / DISC_OBSERVATION_SIZE
    trials = NUM_TRIALS
    trial_len = 500
    dqn_agent = DQ_Net(env=env)
    #dqn_agent.load_model('success-89.h5')


    print("Model loaded")
    state_size = env.observation_space.shape[0]

    for trial in range(trials):
        cur_state = env.reset().reshape(1, state_size)

        cur_disc_state = get_discrete_state(cur_state, env, disc_obs_chunck_size)
        # print(cur_state)
        # print(cur_disc_state)
        for step in range(trial_len):
            if trial % 100 == 0:
                env.render()
            action = dqn_agent.choose_action(cur_disc_state)
            new_state, reward, done, _ = env.step(action)

            new_state = new_state.reshape(1, state_size)
            new_disc_state = get_discrete_state(new_state, env, disc_obs_chunck_size)
            dqn_agent.add_replay_mem(cur_disc_state, action, reward, new_disc_state, done)

            dqn_agent.replay()  # internally iterates default (prediction) model
            dqn_agent.set_target_weights()  # iterates target model

            cur_disc_state = new_disc_state
            if done:
                break
        if step >= 199:
            print("Failed to complete in trial {}, Epsilon is: {}"
                  .format(trial, dqn_agent.epsilon))
            if trial % 10 == 0:
                print("Saving model")
                dqn_agent.save_weights("trial_weigts-{}.h5".format(trial))
                dqn_agent.save_model("trial-{}.h5".format(trial))

        else:
            print("---------------")
            print("Completed in {} trials, , Epsilon is: {}".format(trial, dqn_agent.epsilon))
            print("Step needed", step)
            print("Saving model")
            print("---------------")
            dqn_agent.save_model("success.h5")
            dqn_agent.save_weights('success.h5'.format(trial))


if __name__=="__main__":
    continuous_MC()
    #disc_MC()






