import numpy as np
import oracle

def exploration_rate(time,delta):
    return np.log(405.5*np.power(time, 1.1)/delta)

def get_kl_lower_bound(p,beta, time):
        lo = 0
        hi = p
        q = (lo+hi)/2
        lhs = KL_div(p,q)*time

        while abs(beta-lhs) > 1e-11:
            if abs(hi-lo) < 1e-12:
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

        while abs(beta-lhs) > 1e-11:
            if abs(hi-lo) < 1e-12:
                break
            if lhs > beta:
                hi = q
            else:
                lo = q
            q = (lo+hi)/2
            lhs = KL_div(p,q)*time
        return q

def checkConverged(confidence_bounds):
        for comm in range(confidence_bounds.shape[0]):

        	flag = 0
        	for comm2 in range(confidence_bounds.shape[0]):
        		if(comm==comm2): continue

        		if(confidence_bounds[comm2, 1]> confidence_bounds[comm,0]):
        			flag = 1
        			break

        	if(flag==0): return True

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

def KL_LUCB(numberIterations, randomSeed, populationSize, partyDistribution, deltaValue):

	communityOracle = oracle.communitySampleOracle(randomSeed, populationSize, partyDistribution)
	numSamples = []
	success = []
	numParties = len(partyDistribution)

	# running the experiment numberIterations times

	for iter in range(numberIterations):

		samples = 0.0
		succeed = 0

		commSamples = np.zeros(numParties)
		empiricalMean = np.zeros(numParties)
		confidence_bounds = np.zeros((numParties, 2))


		while(True):

			samples += 1.0
			commSampled = communityOracle.getRandomSample()
			commSamples[commSampled] += 1

			if samples == 1:
				continue

			for comm in range(numParties):
				empiricalMean[comm] = commSamples[comm] / samples

			for comm in range(numParties):
				beta = exploration_rate(samples, deltaValue)
				confidence_bounds[comm, 0] = get_kl_lower_bound(empiricalMean[comm], beta, samples)
				confidence_bounds[comm, 1] = get_kl_upper_bound(empiricalMean[comm], beta, samples)

			shouldExit = False

			for comm in range(numParties):
				if comm == commSampled:
					continue
				if(checkConverged(confidence_bounds)):
					shouldExit = True

			if(shouldExit == False):
				continue

			# We found our answer
			# Check if it is correct

			succeed += communityOracle.predictMode(commSampled)

			break

		numSamples.append(samples)
		success.append(succeed)

		print("iteration " + str(iter) + " num samples " + str(samples) + " success? " + str(succeed))

	return [numSamples, success]
