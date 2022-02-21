import numpy as np
import pandas as pd
from util_ppr2 import *


def randChoice(x):
	
	return np.random.choice(x)

def betaValue(currIterations, communitySamples, numParties, alpha):

	# We calculate the beta value here

	cs = communitySamples
	t = currIterations
	k = numParties
	delta = alpha

	# summation (Zp - Zq)^2 = cs * (cs - 1) / 2
	# cs * (t - cs + 1) / 2

	empiricalVariance = 1.0 / (t * (t - 1)) * (cs * (t - cs))

	logTerm = np.log(4.0 * k *  t * t/ delta)
	term1 = np.sqrt(2.0 * empiricalVariance * logTerm / t)
	term2 = 7.0 * logTerm / (3.0 * (t - 1))
	beta = term1 + term2

	return beta

## LUCB inspired algorithm
			
def runUniformElection_SEEVE(data, alpha, tracefile, batch = 1, init_batch = 1, a = 1, b = 1):

	f = open(tracefile,'w')
	#Gets the names of all the constituencies
	constituencies = getAllPlaces(f"data/{data}/")                              

	#The number of Constituencies
	C = len(constituencies)


	stoppingTimes = np.zeros(C)
	winners = (np.ones(C) * -1).astype(int)

	#List of lists, one list of per party votes for each constituency
	listVotes = []
	#List of lists, one list of party names for each constituency
	listParties = []

	listPartyIDs = []
	
	for c in range(C):
		table = pd.read_csv(f"data/{data}/{constituencies[c]}.csv")
		try:
			table['Votes'] = table['Votes'].str.replace(',','')
		except:
			pass
		#List of votes per party for the constituency c
		votes = list(map(int,table['Votes'].to_list()))
		#List of parties for the constituency c
		parties = table['Party'].to_list()
		
		listVotes.append(votes)
		listParties.append(parties)    
	
	#Gives a unique identity for independent candidates by renaming as Independent0, Independent1...
	indexIndi = 0
	for i, row in enumerate(listParties):
		for j, party in enumerate(row):
			if party == 'Independent':
				listParties[i][j] = f"Independent{indexIndi}"
				indexIndi += 1

				
	#A sorted list(no duplicates) of parties
	Parties = list(set([x for y in listParties for x in y]))
	Parties.sort()

	#Number of parties
	P = len(Parties)
	
	#List of lists, list[i][j] is the id of party j in the list of parties of the ith constituency
	listPartyIDs = [[Parties.index(k) for k in row] for row in listParties]
			
	#Removes the comma and converts the number of votes to integers
	try:
		listVotes = [[int(inner.replace(',', ''))for inner in outer] for outer in listVotes]
	except:
		pass
	unseenVotes = listVotes.copy()
	#Initializes the seen votes to 0 for each candidate in each constituency
	seenVotes = [[0] * len(inner) for inner in listVotes]
	
	hasLost = [[False] * len(inner) for inner in listVotes]

	#Number of constituencies    
	N = C
	
	#Vector of 0s, size = number of political parties - lower bound for number of constituencies won
	Cl = np.zeros(P)
	#Vector with size = number of political parties, each value = number of constituencies - upper bound for number of constituencies won
	Cu = np.zeros(P) 
	countContested = np.zeros(P)
	#Sets Cu[i] to the number of constituencies contested by party i - better upper bound
	for p in range(P):
		countContested[p] = sum(row.count(Parties[p]) for row in listParties)
		Cu[p] = countContested[p]

	# labels of the people accounted
	labels = [[set() for _ in inner] for inner in listVotes]
	votesLabelled = 0

	#The total number of votes (population size) for each constituency
	N0 = [sum(inner) for inner in listVotes]
	
	#Initializes the number of wins to 0 for each party
	seenWins = np.zeros(P)
	seenLosses = np.zeros(P)
	#Adds all the constiuencies to undecided constituencies
	undecidedConstituencies = np.arange(C)

	#Sets the leading party to party number 0 in each constituency as a default
	leadingParty = np.zeros(C)
	while True:
		for c in range(C):
			if c not in undecidedConstituencies:
				continue
			#Number of parties in constituency c
			K = len(listParties[c])

			#Index of the constiuency in the undecided constituency list
			indexC = np.where(undecidedConstituencies == c)
			countWinning = np.array([np.count_nonzero(leadingParty == p) for p in range(P)])

			#Initialization
			for _ in range(init_batch):
				#Calculates the fraction of unseen votes won by each candidate for constiuency c
				norm = [float(i)/sum(unseenVotes[c]) for i in unseenVotes[c]]

				#Samples with probabilities calculated above, returns a one hot vector
				# print(constituencies[c], unseenVotes[c], norm)
				vote = np.random.multinomial(1, norm)

				#Updates the seen and unseen votes accordingly
				# unseenVotes[c] -= vote
				seenVotes[c] += vote

				# updating the labelled information
				party_index = np.argmax(vote)
				person_label = np.random.randint(0, unseenVotes[c][party_index])
				if person_label not in labels[c][party_index]:
					labels[c][party_index].add(person_label)
					votesLabelled += 1

			#Party placed first in c
			constiWinner = np.argsort(seenVotes[c])[-1]

			constiTerm = True
			for p in range(K):
				if p == constiWinner or hasLost[c][p]:
					continue
				mean_w = seenVotes[c][constiWinner] / (seenVotes[c][constiWinner] + seenVotes[c][p])
				mean_p = 1 - mean_w
				beta = betaValue(seenVotes[c][constiWinner] + seenVotes[c][p], seenVotes[c][p], 2, alpha / (C * (K - 1)))
				# partyTerm = checkTerm(alpha/((K - 1)*C), a, b, seenVotes[c][constiWinner] + seenVotes[c][p], seenVotes[c][constiWinner])
				# lcb, ucb = binBounds2(alpha/((K - 1)*C), a, b, seenVotes[c][constiWinner] + seenVotes[c][p], seenVotes[c][constiWinner])
				if mean_p + beta < mean_w - beta:
					hasLost[c][p] = True
					partyID = listPartyIDs[c][p]
					seenLosses[partyID] += 1
				else:
					constiTerm = False

			#Sets the leading party of constituency c to the current winner
			leadingParty[indexC] = Parties.index(listParties[c][constiWinner])

			#Constituency is decided if the lower bound of current winner is above the upper bound of all other parties
			if constiTerm:

				N = N - 1
				winPartyID = int(leadingParty[indexC])

				#Remove the constituency c from the undecided constiuencies and leading party lists
				leadingParty = np.delete(leadingParty, indexC)
				undecidedConstituencies = np.delete(undecidedConstituencies, indexC)

				#Add a win for the winning party
				seenWins[winPartyID] += 1

				#update the lower and upper bounds for each party
				for p in range(P):
					Cl[p] = seenWins[p]
					Cu[p] = countContested[p] - seenLosses[p]

				#winner of the election is the party with the greatest lower bound in number of constituencies won  
				winner = np.argmax(Cl)
				second_place = np.argsort(Cu)[-2]
				second_place_lcb = Cl[second_place]
				second_place_ucb = Cu[second_place]
				#terminate if the lower bound of the leading party is greater than the upper bound of any other party
				term = Cl[winner] - max(Cu[np.arange(len(Cu)) != winner])


				
				print_data = {}
				print_data['constituency_id'] = str(c)
				print_data['constituency'] = str(constituencies[c])
				print_data['consti_winner'] = str(Parties[winPartyID])
				print_data['consti_votes'] = str(sum(seenVotes[c]))
				print_data['total_votes'] = str(sum(map(sum,seenVotes)))
				print_data['undecided_constituencies'] = str(N)
				print_data['winner'] = Parties[winner]
				print_data['winner_lcb'] = str(Cl[winner])
				print_data['winner_ucb'] = str(Cu[winner])
				print_data['second_place'] = Parties[second_place]
				print_data['second_lcb'] = str(second_place_lcb)
				print_data['second_ucb'] = str(second_place_ucb)
				print(print_data)
				f.write(json.dumps(print_data) + '\n')

				#if terminating, print the number of constituencies decided, winning party,
				#constituencies won by the winner, total votes counted		  
				if term > 0:

					totalVotesCounted = sum(map(sum, seenVotes))
					print_data = {}
					print_data['decided_constituencies'] = str(sum(seenWins))
					print_data['winner'] = Parties[winner]
					print_data['seats_won'] = str(seenWins[winner])
					print_data['total_votes'] = str(totalVotesCounted)
					print('Election Complete')
					print(print_data)
					f.write(json.dumps(print_data) + '\n')
					f.close()
					return sum(seenWins), winner, totalVotesCounted, seenVotes, votesLabelled


if __name__ == "__main__":


	alpha = 10**-1
	batch = 100
	init_batch = 100
	data = "India2014"
	
	stoppingC, winner, stoppingVotes, seenVotes = runBanditElectionLUCB(data, alpha, batch, init_batch)

	print(stoppingC, winner, stoppingVotes)
