import numpy as np
import random

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