import copy as c
import numpy as np
import torch as t
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F 

from .network_body import ffBody
from .network_head import Head
from utils import set_requires_grad

class Agent(nn.Module):
	def __init__(self, body_type: 'string | either ff or conv',
					   obs_num_features_or_obs_in_channels: 'int | representing observation dimensions',
					   fc_hidden_layer_size: 'int | number of units in fc hidden layer',
					   output_actions: 'int | actions available',
					   use_target_net: 'Bool | Use target net',
					   g: 'float | Discount rate',
					   lr:'float | learning rate'):
		super(Agent, self).__init__()

		self._body_type = body_type
		self._obs_num_features_or_obs_in_channels = obs_num_features_or_obs_in_channels
		self._fc_hidden_layer_size = fc_hidden_layer_size
		self._output_actions = output_actions
		self._use_target_net = use_target_net
		self._g = g
		self._lr = lr

		self._create_agent_network()
		if self._use_target_net: 
			self._create_target_network()
		self._create_loss()
		self._create_optimiser()

	def _create_agent_network(self):
		if self._body_type == 'ff':
			self._agent_net = Head(ffBody(self._obs_num_features_or_obs_in_channels, self._fc_hidden_layer_size), self._output_actions)
		else:
			self._agent_net = Head(convBody(self._obs_num_features_or_obs_in_channels, self._fc_hidden_layer_size), self._output_actions)

	def _create_target_network(self): 
		if self._body_type == 'ff':
			self._target_net = Head(ffBody(self._obs_num_features_or_obs_in_channels, self._fc_hidden_layer_size), self._output_actions)
		else:
			self._target_net = Head(convBody(self._obs_num_features_or_obs_in_channels, self._fc_hidden_layer_size), self._output_actions)
		set_requires_grad(self._target_net, False)

	def _Q(self, X:'torch tensor batch of observations at time t'):
		return self._agent_net(X)

	def _Q_target(self, X:'torch tensor batch of observations at time t+1'):
		return self._target_net(X)

	def greedy_action(self, X:'torch tensor batch of observations at time t'):
		Q = self._Q(X)
		_, argmax_action = t.max(Q, 1)
		return argmax_action

	def _create_loss(self):
		self._loss = nn.SmoothL1Loss()

	def _create_optimiser(self):
		self._optimiser = optim.Adam(self._agent_net.parameters())

	def loss(self, X:'torch tensor batch of observations at time t',
				   a:'np array batch of actions at time t',
				   r:'np array batch of rewards on transition from t to t+1',
				   X_t:'torch tensor batch of observations at time t+1',
				   T:'np array of terminal status of X_t'):
		# Format a, r, T vectors to (batch_size, 1)
		r = t.from_numpy(r).float().view(-1, 1).float()
		a = t.from_numpy(a).long().view(-1, 1)
		T = t.from_numpy(T).float().view(-1, 1).float()

		# Get Q values for actions a
		Q = self._Q(X)
		Q = Q.gather(1, a)

		# Get values of argmax actions for X_t
		if self._use_target_net:
			Q_target = self._target_net(X_t)
		else:
			Q_target = self._agent_net(X_t).detach()
		Q_target, _ = t.max(Q_target, 1)
		Q_target.unsqueeze_(1)

		# Get TD Error
		T = t.ones_like(T) - T
		Q_target = Q_target * (T)
		Q_target = r + self._g * Q_target  
		return t.mean(self._loss(Q, Q_target))

	def optimise(self, X:'torch tensor batch of observations at time t',
					   a:'np array batch of actions at time t',
					   r:'np array batch of rewards on transition from t to t+1',
					   X_t:'torch tensor batch of observations at time t+1',
					   T:'np array of terminal status of X_t'):
		self._optimiser.zero_grad()
		loss = self.loss(X, a, r, X_t, T)
		loss.backward()
		self._optimiser.step()

	def sync(self):
		self._target_net = c.deepcopy(self._agent_net)