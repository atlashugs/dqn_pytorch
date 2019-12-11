import os
import shutil
import numpy as np
import torch as t

def set_requires_grad(network, grad):
    for param in network.parameters():
        param.requires_grad = grad

def create_exp_logfile(logdir_path):
	if not os.path.exists(logdir_path):
		os.makedirs(logdir_path)
	else:
		shutil.rmtree(logdir_path)
		os.makedirs(logdir_path)

	f_path = logdir_path + '/results.csv'

	return open(f_path, 'w+')

def performance_avg(episodes, k):
	return np.mean(episodes[-k:])