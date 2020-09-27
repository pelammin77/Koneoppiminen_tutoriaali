"""
file: PG_main.py
Author: Petri Lamminaho
Desc:  Policy gradient agent main
creates PG-agent and Gym envirement CartPole game 
"""

import gym
from pg_test import PolicyGradientAgent
import numpy as np

if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    agent = PolicyGradientAgent(ALPHA=0.0005, input_dims=4, GAMMA=0.99,
                                n_actions=2, layer1_size=16, layer2_size=16,
                                chkpt_dir='cart-pole-ckpt')
    # agent.load_model()
    score_history = []
    score = 0
    MAX_EPISODES = 1500
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
        
