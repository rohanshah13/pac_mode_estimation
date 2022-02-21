import numpy as np
import oracle
from math import e
import time

global_delta = -1

def eval_func(delta_val):
	return 2*(e**2) * delta_val*np.exp(-1*delta_val)


def solve_equation(delta):
	left = 0
	right = 700
	mid = int((left+right)/2)

	itera = 0
	while abs( eval_func(mid) - delta) > 1e-300:

		if(abs(right - left)< 1e-7):
			break
		if(delta<eval_func(mid)):
			left = mid
		else:
			right = mid
		mid = (left+right)/2
		itera += 1

	return mid


def exploration_rate(time,new_delta):

	rate = new_delta*(1+np.log(new_delta))*np.log(np.log(time))/((new_delta-1)*np.log(new_delta)) + new_delta

	return rate


def get_kl_lower_bound(p,beta, time):
		lo = 0
		hi = p
		q = (lo+hi)/2
		lhs = KL_div(p,q)*time

		while abs(beta-lhs) > 1e-5:
			if abs(hi-lo) < 1e-7:
				break
			if lhs > beta:
				lo = q
			else:
				hi = q
			q = (lo+hi)/2
			lhs = KL_div(p,q)*time
		return q

def get_kl_upper_bound(p,beta, time):
		lo = p
		hi = 1
		q = (lo+hi)/2
		lhs = KL_div(p,q)*time

		while abs(beta-lhs) > 1e-5:
			if abs(hi-lo) < 1e-7:
				break
			if lhs > beta:
				hi = q
			else:
				lo = q
			q = (lo+hi)/2
			lhs = KL_div(p,q)*time
		return q

def check_termination(p, beta, time):
	lhs = KL_div(p, 0.5)*time
	if lhs > beta:
		return True
	else:
		return False
def checkConverged(confidence_bounds):

		comm = np.argmax(confidence_bounds[:,1])
		flag = 0

		for comm2 in range(confidence_bounds.shape[0]):
			if comm == comm2:
				continue
			if confidence_bounds[comm2, 1] >= confidence_bounds[comm, 0]:
				flag = 1
				break

		if flag == 0:
			return True

		return False

def KL_div(a, b):
		if a == 0:
			if b == 1:
				return float("inf")
			else:
				return (1-a)*np.log((1-a)/(1-b))
		elif a == 1:
			if b == 0:
				return float("inf")
			else:
				return a*np.log(a/b)
		else:
			if b == 0 or b == 1:
				return float("inf")
			else:
				return a*np.log(a/b) + (1-a)*np.log((1-a)/(1-b))

def KL_SN2EVE(numberIterations, randomSeed, populationSize, partyDistribution, deltaValue):

	communityOracle = oracle.communitySampleOracle(randomSeed, populationSize, partyDistribution)
	numSamples = []
	success = []
	time_array = []
	numParties = len(partyDistribution)
	confidence_bounds = np.zeros((numParties, 2))
	new_delta = solve_equation(deltaValue/(numParties-1))

	for iter in range(numberIterations):

		tic = time.perf_counter()

		samples = 0.0
		succeed = 0

		commSamples = np.zeros(numParties)
		empiricalMean = np.zeros(numParties)

		while True:
			sampled_class = communityOracle.getRandomSample()
			commSamples[sampled_class] += 1

			samples += 1
			sampled_class = np.argmax(commSamples)

			# updating ci for each element in the table
			num_victories = 0
			for class_id in range(numParties):
				if class_id == sampled_class:
					continue

				if commSamples[sampled_class] + commSamples[class_id] == 1:
					break

				cond_mean_sampled = commSamples[sampled_class] / (commSamples[sampled_class] + commSamples[class_id]) 
				cond_mean_class = 1 - cond_mean_sampled

				beta_sampled = exploration_rate(commSamples[sampled_class] + commSamples[class_id], new_delta)
				# beta_class = exploration_rate(commSamples[sampled_class] + commSamples[class_id], new_delta)
				# print(beta_sampled)
				# print(beta_class)
				# c_bound11 = get_kl_lower_bound(cond_mean_sampled, beta_sampled, commSamples[sampled_class] + commSamples[class_id])
				# c_bound22 = get_kl_upper_bound(cond_mean_class, beta_class, commSamples[sampled_class] + commSamples[class_id])
				if check_termination(cond_mean_sampled, beta_sampled, commSamples[sampled_class] + commSamples[class_id]): num_victories += 1
				else: break

				# if(c_bound11 > c_bound22): num_victories += 1
				# else: break

			if num_victories == numParties - 1:
				toc = time.perf_counter()
				succeed = communityOracle.predictMode(sampled_class)
				break
		
		time_taken = round(toc - tic, 4)

		numSamples.append(samples)
		success.append(succeed)
		time_array.append(time_taken)

		print("iteration " + str(iter) + " num samples " + str(samples) + " success? " + str(succeed) + " time taken in seconds: " + str(time_taken))

	return [numSamples, success, time_array]