import numpy as np
import oracle
import sys
import time

def betaValue(currIterations, communitySamples, numParties, deltaValue):

	cs = communitySamples
	t = currIterations
	k = numParties
	delta = deltaValue

	# summation (Zp - Zq)^2 = cs * (t - cs)

	empiricalVariance = 1.0 / (t * (t - 1)) * (cs * (t - cs))

	logTerm = np.log(4 * k *  t * t / delta)
	term1 = np.sqrt(2 * empiricalVariance * logTerm / t)
	term2 = 7 * logTerm / (3 * (t - 1))
	beta = term1 + term2

	return beta

def sequentialEstimation(numberIterations, randomSeed, populationSize, partyDistribution, deltaValue):

	communityOracle = oracle.communitySampleOracle(randomSeed, populationSize, partyDistribution)
	numSamples = []
	success = []
	time_array = []
	numParties = len(partyDistribution)

	# running the experiment numberIterations times

	for iter in range(numberIterations):

		tic = time.perf_counter()

		samples = 0.0
		succeed = 0

		commSamples = np.zeros(numParties)
		empiricalMean = np.zeros(numParties)


		while(True):

			samples += 1.0
			commSampled = communityOracle.getRandomSample()
			commSamples[commSampled] += 1

			if samples == 1:
				continue

			for comm in range(numParties):
				empiricalMean[comm] = commSamples[comm] / samples

			ind = np.argmax(empiricalMean)

			beta1 = betaValue(samples, commSamples[ind], numParties, deltaValue)

			shouldExit = True

			for comm in range(numParties):
				if comm == commSampled:
					continue
				betaComm = betaValue(samples, commSamples[comm], numParties, deltaValue)
				if (empiricalMean[comm] + betaComm) >= (empiricalMean[ind] - beta1):
					shouldExit = False
					break

			if shouldExit == False:
				continue

			toc = time.perf_counter()
			succeed += communityOracle.predictMode(ind)

			break

		time_taken = round(toc - tic, 4)

		numSamples.append(samples)
		success.append(succeed)
		time_array.append(time_taken)

		print("iteration " + str(iter) + " num samples " + str(samples) + " success? " + str(succeed) + " time taken in seconds: " +str(time_taken))

	return [numSamples, success, time_array]