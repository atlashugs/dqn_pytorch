import numpy as np
import random

from collections import deque

class ExperienceReplayBuffer:
	def __init__(self, size:'int | maximum size of er buffer', 
					   batch_size:'int | size of mini-batch training samples'):
		self.size = size
		self.batch_size = batch_size
		self.buffer = deque(maxlen=self.size)

	def append(self, s, a, r, n_s, t):
		s = np.expand_dims(s, 0)
		n_s = np.expand_dims(n_s, 0)
		self.buffer.append((s, a , r, n_s, t))

	def sample(self):
		s, a, r, n_s, t = zip(*random.sample(self.buffer, self.batch_size))
		return np.concatenate(s), a, r, np.concatenate(n_s), t

	def __len__(self):
		return len(self.buffer)