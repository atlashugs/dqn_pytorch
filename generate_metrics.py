import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

RUNS = 10
EVALUATIONS = 75

def plot_curves(directory):
	results_dirs = os.listdir(directory)
	results = np.zeros((RUNS, EVALUATIONS)) 
	for alpha_directory in results_dirs:
		try:
			for run in range(RUNS):
				run_results_file = os.path.join(directory, alpha_directory, str(run), 'results.csv')
				run_results = np.genfromtxt(run_results_file, delimiter=',')
				results[run, :] = run_results[:EVALUATIONS, 1]
			plt.plot(np.mean(results, axis=0), label=alpha_directory)
		except:
			continue
	plt.legend()
	plt.show()

def auc(directory):
	results_dirs = os.listdir(directory)
	results = np.zeros((RUNS, EVALUATIONS)) 
	for alpha_directory in results_dirs:
		try:
			for run in range(RUNS):
				run_results_file = os.path.join(directory, alpha_directory, str(run), 'results.csv')
				run_results = np.genfromtxt(run_results_file, delimiter=',')
				results[run, :] = run_results[:EVALUATIONS, 1]
			aucs = np.trapz(results, axis=1, dx=1)
		except:
			continue
		print('Learning Rate: ', alpha_directory, 'Mean: ', np.mean(aucs), 'Std: ', np.std(aucs))

def auc_summarize_all_algorithms(base_dir):
	algorithm_results = {}
	algorithm_results['DQN'] = 'DQN/7.5e-05/'
	algorithm_results['Gamma_Scale'] = 'gamma_scale_DQN/5e-05/'
	algorithm_results['Multihead'] = 'multihead_DQN/5e-05/'
	algorithm_results['Bandit'] = 'bandit_DQN/7.5e-05/'

	algs = []
	alg_means = []
	alg_std_dev = []
	for algorithm_name, algo_alpha_dir in algorithm_results.items():
		algs.append(algorithm_name)
		results = np.zeros((RUNS, EVALUATIONS)) 
		run_dirs = 	os.listdir(os.path.join(base_dir, algo_alpha_dir))
		for run, run_dir in enumerate(run_dirs):
			run_results_file = os.path.join(base_dir, algo_alpha_dir, run_dir, 'results.csv')
			run_results = np.genfromtxt(run_results_file, delimiter=',')
			results[run, :] = run_results[:EVALUATIONS, 1]
		aucs = np.trapz(results, axis=1, dx=1)
		alg_means.append(np.mean(aucs))
		alg_std_dev.append(np.std(aucs) / np.sqrt(RUNS))



	print(algs)
	print(alg_means)
	print(alg_std_dev)


	plt.errorbar([algs[0]], [alg_means[0]], yerr=[alg_std_dev[0]], fmt='o', color='blue')
	plt.errorbar([algs[1]], [alg_means[1]], yerr=[alg_std_dev[1]], fmt='o', color='red')
	plt.show()


if __name__ == '__main__':
	# auc_summarize_all_algorithms('./CARTPOLE_RESULTS')
	print('DQN')
	auc('./LUNARLANDER_RESULTS/DQN/')
	print('Gamma Scale DQN')
	auc('./LUNARLANDER_RESULTS/gamma_scale_DQN/')
	print('Multihead DQN')
	auc('./LUNARLANDER_RESULTS/Multihead_DQN/')
	print('Bandit DQN')
	auc('./LUNARLANDER_RESULTS/bandit_DQN/')