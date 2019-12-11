import numpy as np 
import torch as t
import torch.nn as nn 
import torch.nn.functional as F

class Head(nn.Module):
	def __init__(self, body: 'torch | object representing network body',
					   output_actions: 'int | actions available'):
		super(Head, self).__init__()
		self._body = body
		self._output_actions = output_actions

		self._create_head()

	def _create_head(self):
		self._network = nn.Sequential(nn.Linear(in_features=self._body._fc_hidden_layer_size, out_features=self._output_actions))

	def forward(self, X):
		X = self._body(X)
		return self._network(X).view(-1, self._output_actions)