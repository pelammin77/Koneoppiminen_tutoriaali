"""
file: MC_q_learning.py
Author: Petri Lamminaho
Desc: Q-Learning agent plays Gym's Mountain Car game
"""
import gym
import numpy as np
import matplotlib.pyplot as plt
##################################################
LEARNING_RATE  = 0.1
DISCOUNT  = 0.95
episodes = 1000000
RENDER_STEP = 10000
STAT_REWARD = 5000
epsilon = 0.6
START_EPSILON_DECAY = 1
END_EPSILON_DECAY = episodes // 2
epsilon_decay_value  = epsilon / (END_EPSILON_DECAY - START_EPSILON_DECAY)
env = gym.make('MountainCar-v0')
DISC_OBSERVATION_SIZE = [20] * len(env.observation_space.low)
disc_obs_chunck_size = (env.observation_space.high - env.observation_space.low) / DISC_OBSERVATION_SIZE
q_table = np.random.uniform(low=-2, high=0, size=(DISC_OBSERVATION_SIZE + [env.action_space.n]))
episode_reward_list = []
rewards_info ={'episode':[], 'min':[], 'max':[], 'avg':[]}
##############################################################

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / disc_obs_chunck_size
    return tuple (discrete_state.astype(np.int))

if __name__ == '__main__':
    for episode in range(episodes):
        if episode % 10 == 0:
            np.save("qtables/"+ str(episode)+"-qtable.npy", q_table)
        episode_reward = 0
        discrete_state = get_discrete_state(env.reset())
        done = False
        if episode % RENDER_STEP == 0:
            render = True
            print(episode)
        else:
            render = False
        while not done:
            if np.random.random() > epsilon:
                action = np.argmax(q_table[discrete_state])
            else:
                action = np.random.randint(0, env.action_space.n)
            new_sate, reward, done, info = env.step(action)
            episode_reward += reward
            new_discrete_state = get_discrete_state(new_sate)
            if render==True:
                env.render()

            if not done:
                max_future_q = np.max(q_table[new_discrete_state])
                current_q = q_table[discrete_state + (action, )]
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
                q_table[discrete_state + (action, )] = new_q

            elif new_sate[0] >= env.goal_position:
                q_table[discrete_state + (action, )] = reward
                q_table[discrete_state + (action, )] = 0
                #print("Task solved episode", episode)
            discrete_state = new_discrete_state
        if END_EPSILON_DECAY >= episode >= START_EPSILON_DECAY:
            epsilon -= epsilon_decay_value

        episode_reward_list.append(episode_reward)
        if episode % STAT_REWARD == 0:
            avg_reward = sum(episode_reward_list[-STAT_REWARD:])/len(episode_reward_list[-STAT_REWARD:])
            min_reward = min(episode_reward_list[-STAT_REWARD:])
            max_reward = max(episode_reward_list[-STAT_REWARD:])
            rewards_info['episode'].append(episode)
            rewards_info['avg'].append(avg_reward)
            rewards_info['min'].append(min_reward)
            rewards_info['max'].append(max_reward)

            print("Episode", episode, "Worst:", min_reward, "Best:", max_reward, "Avg reward", avg_reward)
    plt.plot(rewards_info['episode'], rewards_info['min'], Label="Worst")
    plt.plot(rewards_info['episode'], rewards_info['max'], Label="Best")
    plt.plot(rewards_info['episode'], rewards_info['avg'], Label="Avg")
    plt.legend(loc=4)
    plt.show()
    env.close()

