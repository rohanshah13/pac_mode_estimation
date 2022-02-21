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

def sequentialEstimationEVE(numberIterations, randomSeed, populationSize, partyDistribution, deltaValue):

    communityOracle = oracle.communitySampleOracle(randomSeed, populationSize, partyDistribution)
    numSamples = []
    success = []
    time_array = []
    numParties = len(partyDistribution)

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

            num_victories = 0
            for class_id in range(numParties):
                if class_id == sampled_class:
                    continue

                if commSamples[sampled_class] + commSamples[class_id] == 1:
                    break

                cond_mean_sampled = commSamples[sampled_class] / (commSamples[sampled_class] + commSamples[class_id]) 
                cond_mean_class = 1 - cond_mean_sampled
                beta_sampled = betaValue(
                    commSamples[sampled_class] + commSamples[class_id], commSamples[sampled_class], 2, deltaValue / (numParties - 1.0)
                )
                beta_class = betaValue(
                    commSamples[sampled_class] + commSamples[class_id], commSamples[class_id], 2, deltaValue / (numParties - 1.0)
                )

                if (cond_mean_sampled - beta_sampled > cond_mean_class + beta_class): num_victories += 1
                else: break

            if num_victories == numParties - 1:
                toc = time.perf_counter()
                succeed = communityOracle.predictMode(sampled_class)
                break
        
        time_taken = round(toc - tic, 4)

        numSamples.append(samples)
        success.append(succeed)
        time_array.append(time_taken)

        print("iteration " + str(iter) + " num samples " + str(samples) + " success? " + str(succeed) + " time taken in seconds: " +str(time_taken))

    return [numSamples, success, time_array]