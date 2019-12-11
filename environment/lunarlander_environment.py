#!/bin/python3

import numpy as np
import gym

class LunarLander:
	def __init__(self):
		self._env = gym.make('LunarLander-v2')

	def reset(self):
		self._cum_reward = 0 

		return self._env.reset()

	def act(self, action):	
		observation, reward, terminal, _ = self._env.step(action)
		self._cum_reward += reward

		return reward, observation, terminal

	def close(self):
		return

	def action_set(self):
		return [0, 1, 2, 3]

	def num_actions(self):
		return len([0, 1, 2, 3])

	def episode_return(self):
		return self._cum_reward