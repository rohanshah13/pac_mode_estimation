import os
import copy
import math
import numpy as np
import argparse
from scipy.stats import beta
import sys
import time

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from util import write_data

class communitySampleOracle:

	# population size is total number of people, for eg. 1000
	# partyDistribution is how the people are distributed among parties in a list
	# For eg, if population size = 1000, partyDistribution may be [350 650]
	# The largest party should be unique in this code

	def __init__(self, randomSeed, populationSize, partyDistribution):
		
		np.random.seed(randomSeed)
		self.populationSize = populationSize
		self.numParties = len(partyDistribution)
		self.distribution = [(x * 1.0) / populationSize for x in partyDistribution]
		return

	def getRandomSample(self):

		communityPulled = np.random.choice(self.numParties, 1, p = self.distribution)
		return communityPulled[0]

	def predictMode(self, prediction):

		modeValue = max(self.distribution)
		actualMode = self.distribution.index(modeValue)
		return prediction == actualMode


# function for finding the confidence sequence at a particular t in horizon
def binary_confidence_sequence(params_init, params_final, step_size=0.01, alpha=0.05, e=1e-12):
	'''
		params_init: parameters of the prior beta distribution (list)
		params_final: parameters of the posterior beta distribution (list)
		step_size: For searching the parameter space
		alpha: error probability
	'''

	# We implement PPR-1vr by dividing our parameter space into 1/step_size parts and maintaining confidence sequences
	# This can be done more accurately by ternary searching a point in the confidence sequence (say x), and then 
	# binary searching in [0, x] and [x, 1] for the bounds of the confidence sequence
	# However, the above is slower than this approach

	# possible p values
	p_vals = np.linspace(0, 1, num=int(1 / step_size) + 1)
	indices = np.arange(len(p_vals))

	# computation of prior
	log_prior_0 = beta.logpdf(p_vals, params_init[0], params_init[1])
	log_prior_1 = beta.logpdf(p_vals, params_init[1], params_init[0])

	# computation of posterior
	log_posterior_0 = beta.logpdf(p_vals, params_final[0], params_final[1])
	log_posterior_1 = beta.logpdf(p_vals, params_final[1], params_final[0])

	# martingale computation
	log_martingale_0 = log_prior_0 - log_posterior_0
	log_martingale_1 = log_prior_1 - log_posterior_1

	# Confidence intervals
	ci_condition_0 = log_martingale_0 < np.log(1 / alpha)
	ci_condition_1 = log_martingale_1 < np.log(1 / alpha)
	
	ci_indices_0 = np.copy(indices[ci_condition_0])
	ci_indices_1 = np.copy(indices[ci_condition_1])
	return [p_vals[np.min(ci_indices_0)], p_vals[np.max(ci_indices_0)]], [p_vals[np.min(ci_indices_1)], p_vals[np.max(ci_indices_1)]]


# function for determining the number of samples when multiple parties are present
def onevsall_ppr(oracle, params_init, step_size=0.01, alpha=0.05):
	'''
		oracle: object of the type communitySampleOracle with len(party_distribution) = 2
		params_init: list of parameters of the prior beta distribution for each part [[1, 1], [1, 1], ... [1, 1]]
		step_size: For searching the parameter space
		alpha: error probability
	'''

	# checking if the oracle and params_init is consistent
	assert(len(params_init) == len(oracle.distribution))
	
	tic = time.perf_counter()

	n = len(oracle.distribution)

	params_final = copy.deepcopy(params_init)
	p_final = copy.deepcopy(params_init)
	ci_list = copy.deepcopy(params_init)

	winning = np.zeros(n)

	largest_val = 0
	large_ind = 0

	t = 0
	while True:
		sampled_class = oracle.getRandomSample()

		params_final[sampled_class][0] += 1

		if(params_final[sampled_class][0] >= largest_val):
			largest_val = params_final[sampled_class][0]
			large_ind = sampled_class

		while True:
			if(winning[large_ind] == large_ind):
				winning[large_ind] += 1
				continue
			if(winning[large_ind] >= n):
				toc = time.perf_counter()
				time_taken = round(toc - tic, 4)
				return [oracle.predictMode(large_ind), t+1, time_taken]
			
			compete = int(winning[large_ind])

			p_final[large_ind][0] = params_final[large_ind][0]
			p_final[large_ind][1] = t + 1 + 2 * params_init[large_ind][0] - params_final[large_ind][0]

			ci_list[large_ind], _ = binary_confidence_sequence(params_init[large_ind], p_final[large_ind],
								   step_size=step_size, alpha=alpha / len(oracle.distribution))

			p_final[compete][0] = params_final[compete][0]
			p_final[compete][1] = t + 1 + 2 * params_init[large_ind][0] - params_final[compete][0]

			ci_list[compete], _ = binary_confidence_sequence(params_init[compete], p_final[compete],
								   step_size=step_size, alpha=alpha / len(oracle.distribution))


			if ci_list[large_ind][0] > ci_list[compete][1]:
				winning[large_ind] += 1
			else:
				break
		t = t + 1

# function for evaluating the PPR model for a number of iterations
def ppr_evaluate(oracle, params_init, step_size=0.01, alpha=0.05, iterations=100, log=True):
	'''
		oracle: object of the type communitySampleOracle
		params_init: parameters of the prior beta distribution (list)
		step_size: For searching the parameter space
		alpha: error probability
		iterations: Number of times the algorithm has to evaluated
		log: Set False to suppress logs
	'''

	# variables to be returned
	num_samples = []
	success = []
	time_array = []

	for i in range(iterations):
		sys.stderr.write("Iteration " + str(i) + '\n')
		[success_bool, time_index, time_taken] = onevsall_ppr(oracle, params_init, step_size, alpha)
		num_samples.append(time_index)
		success.append(int(success_bool))
		time_array.append(time_taken)
		if log:
			print('Iteration: {}, Samples required: {}, success: {}, time taken in seconds: {}'.format(i, time_index, success_bool, time_taken))

	return [num_samples, success, time_array]


if __name__ == '__main__':

	# for parsing the arguments
	parser = argparse.ArgumentParser(description='PPR method for mode estimation for sampling with replacement')
	parser.add_argument('--random_seed', '-rs', type=int, default=1)
	parser.add_argument('--population_size', '-ps', type=int, default=1000)
	parser.add_argument('--party_distribution', '-pd', type=str, default='900_100', help='Input the list of numbers separated by "_"')
	parser.add_argument('--step_size', '-ss', type=float, default=0.01, help='granularity of search space while solving for PPR < 1 / delta')
	parser.add_argument('--delta', '-d', type=float, default=0.01, help='Probability of error')
	parser.add_argument('--iterations', '-i', type=int, default=100, help='number of experiments to be executed')
	parser.add_argument('--outfile', default='data.txt')
	args = parser.parse_args()

	# obtaining the list of numbers in appropriate format
	party_distribution = [float(num) for num in args.party_distribution.split('_')]

	# initializing oracle
	oracle = communitySampleOracle(args.random_seed, args.population_size, party_distribution)

	# Initializing the parameters
	params_init = [[1, 1] for index in range(len(party_distribution))]

	[num_samples, num_success, time_taken] = ppr_evaluate(oracle, params_init, args.step_size, args.delta, args.iterations)
	
	std = np.std(num_samples)
	std_time = np.std(time_taken)

	num_samples = sum(num_samples) / args.iterations
	num_success = sum(num_success) / args.iterations
	avg_time = sum(time_taken) / args.iterations

	se = std/np.sqrt(args.iterations)
	se_time = std_time/np.sqrt(args.iterations)

	failure_probability = 1 - num_success

	print("Failure probability is "+str(1-num_success))
	print("Average number of samples are "+str(num_samples))
	print("Standard error is "+str(round(se, 2)))
	print("Average time taken is "+str(avg_time))
	print("Standard error for time is "+str(se_time))

	write_data('PPR-Bernoulli-1vr',num_samples, se, args.delta, failure_probability, args.random_seed, args.outfile)
