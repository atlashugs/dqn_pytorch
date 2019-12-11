import numpy as np 
import torch as t
import torch.nn as nn 
import torch.nn.functional as F 

class ffBody(nn.Module):
	def __init__(self, obs_num_features: 'int | number of features in observation vector',
					   fc_hidden_layer_size: 'int | number of units in fc hidden layer'):
		super(ffBody, self).__init__()
		self._obs_num_features = obs_num_features
		self._fc_hidden_layer_size = fc_hidden_layer_size

		self._create_body()

	def _create_body(self):		
		self._network = nn.Sequential(nn.Linear(in_features=self._obs_num_features, out_features=self._fc_hidden_layer_size), 
									  nn.ReLU())

	def forward(self, X):
		return self._network(X)