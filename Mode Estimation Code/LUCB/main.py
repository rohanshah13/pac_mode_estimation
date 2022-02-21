import hoeffding
import numpy as np
import sys
import argparse
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from util import write_data

parser = argparse.ArgumentParser(description='A1-1vr method for mode estimation for sampling with replacement')
parser.add_argument('--random_seed', '-rs', type=int, default=0)
parser.add_argument('--population_size', '-ps', type=int, default=1000)
parser.add_argument('--party_distribution', '-pd', type=str, default='900_50_50', help='Input the list of numbers separated by "_"')
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

[numSamples, success] = hoeffding.Hoeffding(numberIterations, randomSeed,
	populationSize, partyDistribution, deltaValue)

avgSamples = sum(numSamples) / len(numSamples)
avgSuccess = sum(success) / len(success)
avgError = 1.0 - avgSuccess

numSamples = np.asarray(numSamples)
std = np.std(numSamples)
se = std/np.sqrt(numberIterations)

failure_probability = 1 - avgSuccess

print("Average samples " + str(round(avgSamples, 3)))
print("Standard Error " + str(round(se, 2)))
print("Average error " + str(round(avgError, 3)))
print("Expected error " + str(deltaValue))

write_data('LUCB',avgSamples, se, args.delta, failure_probability, args.random_seed, args.outfile)
