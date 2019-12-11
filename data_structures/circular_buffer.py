import numpy as np 
import torch as t

from collections import deque

class CircularBuffer:
	def __init__(self, size:'int | history length'):
		self.size = size
		self.history = deque(maxlen=self.size)

	def append(self, f: 'torch.tensor | most recent observation frame'):
		f = f.unsqueeze(0)
		self.history.append((f))

	def __call__(self):
		return t.from_numpy(np.concatenate(self.history, axis=0))