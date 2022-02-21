import KLSN2EVE
import numpy as np
import os
import sys
import argparse

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from util import write_data

parser = argparse.ArgumentParser(description='KL-SN-1v1 method for mode estimation for sampling with replacement')
parser.add_argument('--random_seed', '-rs', type=int, default=1)
parser.add_argument('--population_size', '-ps', type=int, default=1000)
parser.add_argument('--party_distribution', '-pd', type=str, default='950_50', help='Input the list of numbers separated by "_"')
parser.add_argument('--delta', '-d', type=float, default=0.01)
parser.add_argument('--iterations', '-i', type=int, default=100)
parser.add_argument('--outfile', default='data.txt')
args = parser.parse_args()

randomSeed = args.random_seed
numberIterations = args.iterations
populationSize = args.population_size

partyDistribution = [float(num) for num in args.party_distribution.split('_')]

deltaValue = args.delta

if sum(partyDistribution) != populationSize:
	raise Exception("partyDistribution does not add up to populationSize")

[numSamples, success, time_taken] = KLSN2EVE.KL_SN2EVE(numberIterations, randomSeed,
	populationSize, partyDistribution, deltaValue)

avgSamples = sum(numSamples) / len(numSamples)
avgSuccess = sum(success) / len(success)
avgError = 1.0 - avgSuccess
avg_time = sum(time_taken) / len(time_taken)

std = np.std(numSamples)
std_time = np.std(time_taken)

se = std / np.sqrt(numberIterations)
se_time = std_time/np.sqrt(numberIterations)

failure_probability = 1 - avgSuccess

print("Average samples " + str(round(avgSamples, 3)))
print("Average error " + str(round(avgError, 3)))
print("Expected error " + str(deltaValue))
print("Standard error " + str(se))
print("Average time taken is "+str(avg_time))
print("Standard error for time is "+str(se_time))

write_data('KLSN-1v1',avgSamples, se, args.delta, failure_probability, args.random_seed, args.outfile)
