import os
import numpy as np 
import torch as t
t.set_printoptions(threshold=5000)

from absl import app
from absl import flags

from dqn import Agent
from data_structures import CircularBuffer, ExperienceReplayBuffer
from environment import LunarLander
from utils import create_exp_logfile, performance_avg

FLAGS = flags.FLAGS

# RL flags
flags.DEFINE_float('init_epsilon', 1, 'Random action selection starting probability')
flags.DEFINE_float('final_epsilon', 0.01, 'Random action selection ending probability')
flags.DEFINE_float('epsilon_anneal', 50000, 'Epsilon anneal schedule')
flags.DEFINE_float('gamma', 0.99, 'Discount parameter')

# Network flags
flags.DEFINE_string('network_type', 'ff', 'Either ff or conv indicating type of input')
flags.DEFINE_integer('observation_dimensions', 8, 'Dimensions of input observation')
flags.DEFINE_integer('fc_hidden_layer_size', 32, 'Number of units to use in fully-connected hidden layer')
flags.DEFINE_integer('update_frequency', 4, 'Frequency of value fn updates')
flags.DEFINE_boolean('use_target_net', True, 'Use target network')
flags.DEFINE_integer('target_network_update', 800, 'Frequency to update target network')
flags.DEFINE_float('learning_rate', 0.00025, 'Learning rate for optimiser')

# Experiment flags
flags.DEFINE_integer('seed', 0, 'Random seed')
flags.DEFINE_integer('max_steps', 100000, 'Maximum number of steps to run')
flags.DEFINE_string('exp_log_dir', './LUNARLANDER_RESULTS/DQN/', 'Directory to store experiment logs')
flags.DEFINE_integer('evaluate', 1000, 'Evaluation frequency')
flags.DEFINE_integer('num_episodes_average', 10, 'Completed episodes to average for performance evaluation')

# ER structures flags
flags.DEFINE_integer('er_size', 10000, 'Capacity of Experience Replay buffer')
flags.DEFINE_integer('batch_size', 32, 'Size of sample batch')

# Frame History flags
flags.DEFINE_integer('cb_size', 1, 'History length of Circular Buffer')

def main(argv):
	# Set seeds
	np.random.seed(FLAGS.seed)
	t.manual_seed(FLAGS.seed)

	# Create logfile
	f = create_exp_logfile(os.path.join(FLAGS.exp_log_dir, str(FLAGS.learning_rate), str(FLAGS.seed)))

	# Initialise agent and environment
	env = LunarLander()
	# env = Gridworld()
	num_actions = env.num_actions()
	agent = Agent(body_type='ff', 
				  obs_num_features_or_obs_in_channels=FLAGS.observation_dimensions, 
				  fc_hidden_layer_size = FLAGS.fc_hidden_layer_size, 
				  output_actions = num_actions, 
				  use_target_net = FLAGS.use_target_net,
				  g = FLAGS.gamma, 
				  lr = FLAGS.learning_rate)

	# Initialise data structures
	c_buf = CircularBuffer(size=FLAGS.cb_size)
	er_buf = ExperienceReplayBuffer(size=FLAGS.er_size, batch_size=FLAGS.batch_size)

	# Initialise sampling range for e-greedy
	interval = t.distributions.uniform.Uniform(t.tensor([0.0]), t.tensor([1.0]))

	# Run
	step = 0
	episode_results = []
	state = env.reset()
	c_buf.append(t.from_numpy(state).float())

	while step < FLAGS.max_steps:
		# Agent select action
		eps = max(FLAGS.init_epsilon - (((FLAGS.init_epsilon - FLAGS.final_epsilon) / FLAGS.epsilon_anneal) * step), FLAGS.final_epsilon)

		if interval.sample() <= eps:
			action = np.random.randint(num_actions)
		else:
			action = agent.greedy_action(c_buf()).item()
		reward, next_state, terminal = env.act(action)
		terminal = 1 if terminal else 0

		er_buf.append(state, action, reward, next_state, terminal)
		state = next_state
		c_buf.append(t.from_numpy(state).float())

		if step > FLAGS.batch_size and step % FLAGS.update_frequency:
			batch_states, batch_actions, batch_rewards, batch_next_states, batch_terminals = \
																				er_buf.sample()
			batch_states = t.from_numpy(batch_states).float()
			batch_actions = np.array(batch_actions)
			batch_rewards = np.array(batch_rewards)
			batch_next_states = t.from_numpy(batch_next_states).float()
			batch_terminals = np.array(batch_terminals)

			agent.optimise(batch_states, batch_actions, batch_rewards, batch_next_states, batch_terminals)

		if step % FLAGS.target_network_update == 0:
			agent.sync()

		if terminal:
			episode_results.append(env.episode_return())
			state = env.reset()
			
		step += 1
	
		if step % FLAGS.evaluate == 0:
			f.write('{}, {}\n'.format(step, performance_avg(episode_results, FLAGS.num_episodes_average)))
			f.flush()

	f.close()

if __name__ == '__main__':
	app.run(main)

