import os
import copy
import math
import numpy as np
import argparse
from scipy.stats import beta
import sys
import time
import json

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

LIM = int(1e6)
log_fact = np.zeros(LIM)
log_pow_2 = np.zeros(LIM)

# Our own implementation of scipy.stats.beta.logpdf, when evaluated at 0.5
def compute_beta_half(x, y):
    x = int(x)
    y = int(y)
    ans = log_pow_2[x+y-2]
    ans -= log_fact[x-1]
    ans -= log_fact[y-1]
    ans += log_fact[x+y-1]
    return ans


def ppr_me(oracle, alpha=0.05):

    tic = time.perf_counter()

    n = len(oracle.distribution)
    separated = np.zeros((n, n))

    alpha = alpha / (n - 1.0)

    sample_table = np.ones(n)
    t = 0

    largest_val = np.zeros(2, dtype='int')
    large_ind = np.zeros(2, dtype='int')
    large_ind[1] = 1

    while True:
        sampled_class = oracle.getRandomSample()
        sample_table[sampled_class] += 1
        t += 1

        if(sample_table[sampled_class] >= largest_val[0]):
            if(large_ind[0] != sampled_class):
                largest_val[1] = largest_val[0]
                large_ind[1] = large_ind[0]
                largest_val[0]=  sample_table[sampled_class]
                large_ind[0] = sampled_class
            else:
                largest_val[0] = sample_table[sampled_class]

        elif(sample_table[sampled_class] >= largest_val[1]):
            largest_val[1] = sample_table[sampled_class]
            large_ind[1] = sampled_class

        # below uses our implementation - much faster
        log_posterior = compute_beta_half(sample_table[large_ind[0]], sample_table[large_ind[1]])
        # scipy version - log_posterior = beta.logpdf(0.5, sample_table[large_ind[0]], sample_table[large_ind[1]])

        log_alpha = np.log(1/alpha)

        if(-log_posterior >= log_alpha):
            toc = time.perf_counter()
            time_taken = round(toc - tic, 4)
            return [oracle.predictMode(sampled_class), t, time_taken]

def ppr_evaluate(oracle, alpha=0.05, iterations=100, log=True):
    '''
        oracle: object of the type communitySampleOracle
        alpha: error probability
        iterations: Number of times the algorithm has to evaluated
        log: Set False to suppress logs
    '''

    num_samples = []
    success = []
    time_array = []

    for i in range(iterations):
        sys.stderr.write("Iteration " + str(i) + "\n")
        [success_bool, time_index, time_taken] = ppr_me(oracle, alpha)
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
    parser.add_argument('--party_distribution', '-pd', type=str, default='950_50', help='Input the list of numbers separated by "_"')
    parser.add_argument('--delta', '-d', type=float, default=0.01)
    parser.add_argument('--iterations', '-i', type=int, default=100)
    parser.add_argument('--outfile', default='data.txt')
    args = parser.parse_args()

    for i in range(LIM):
        log_pow_2[i] = np.log(0.5) * i

    log_fact[0] = 0
    for i in range(1, LIM):
        log_fact[i] = log_fact[i-1] + np.log(i)

    # obtaining the list of numbers in appropriate format
    party_distribution = [float(num) for num in args.party_distribution.split('_')]

    oracle = communitySampleOracle(args.random_seed, args.population_size, party_distribution)

    [num_samples, num_success, time_taken] = ppr_evaluate(oracle, args.delta, args.iterations)    

    std = np.std(num_samples)
    std_time = np.std(time_taken)

    num_samples = sum(num_samples) / args.iterations
    num_success = sum(num_success) / args.iterations
    avg_time = sum(time_taken) / args.iterations

    se = std/np.sqrt(args.iterations)
    se_time = std_time/np.sqrt(args.iterations)

    failure_probability = str(1 - num_success)

    print("Failure probability is "+str(1-num_success))
    print("Average number of samples are "+str(num_samples))
    print("Standard error is "+str(round(se, 2)))
    print("Average time taken is "+str(avg_time))
    print("Standard error for time is "+str(se_time))
    
    write_data('PPR-Bernoulli-1v1',num_samples, se, args.delta, failure_probability, args.random_seed, args.outfile)