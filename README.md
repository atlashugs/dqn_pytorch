# DQN for vector inputs
A simple, efficient, and easily extensible implementation of DQN designed for environments where the input to the network is a feature vector (e.g. Cartpole, Catcher, LunarLander).

# Dependencies
This code is developed on Python3 running on Ubuntu 18.04. It uses:
* Pytorch 1.1.0
* Numpy 1.16.3
* Absl-py 0.7.1
* Gym 0.12.1

# Usage
0. Install all dependencies
1. Clone/download this repository
2. Modify ```environments/__init__.py``` to be whatever environment you want to run on (you can add your own here too!)
3. Modify lines ```11, 54```  in ```experiment/run_DQN.py``` to appropriately match the environment you chose
4. Use whatever parameters you wish in the ```flags``` variables in ```experiment/run_DQN.py```
5. Navigate to the base directory (i.e. ```dqn_pytorch```) and enter ```python3 -m experiment.run_DQN```
