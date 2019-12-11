#!/bin/python3

import numpy as np
import gym

class CartPole:
	def __init__(self):
		self._env = gym.make('CartPole-v0')

	def reset(self):
		self._cum_reward = 0 

		return self._env.reset()

	def act(self, action):	
		observation, reward, terminal, _ = self._env.step(action)
		
		# reward = reward if not terminal else -reward
		self._cum_reward += reward

		return reward, observation, terminal

	def close(self):
		return

	def action_set(self):
		return [0, 1]

	def num_actions(self):
		return len([0, 1])

	def episode_return(self):
		return self._cum_reward