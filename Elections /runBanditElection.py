import numpy as np
import pandas as pd
from util import *
import json


def randChoice(x):
	
	return np.random.choice(x)

## LUCB inspired algorithm
			
def runBanditElectionLUCB(data, alpha, tracefile, batch = 1, init_batch = 1, a = 1, b = 1):
	TMP = 'tmp.txt'
	f = open(tracefile, 'w')
	tempfile = open(TMP, 'w')
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
	
	#Number of constituencies    
	N = C
	
	#Vector of 0s, size = number of political parties - lower bound for number of constituencies won
	Cl = np.zeros(P)
	#Vector with size = number of political parties, each value = number of constituencies - upper bound for number of constituencies won
	Cu = np.ones(P) * C

	#Sets Cu[i] to the number of constituencies contested by party i - better upper bound
	for p in range(P):
		
		Cu[p] = sum(row.count(Parties[p]) for row in listParties)

	#The total number of votes (population size) for each constituency
	N0 = [sum(inner) for inner in listVotes]
	#Initalizes votes to 0 for each candidate in each constituency - lower bound
	Nl = [[0] * len(inner) for inner in listVotes]
	#Initializes votes to max possible for each candidate in each constituency - upper bound
	Nu = [[sum(inner)] * len(inner) for inner in listVotes]
	#Initializes the number of wins to 0 for each party
	seenWins = np.zeros(P)

	#Adds all the constiuencies to undecided constituencies
	undecidedConstituencies = np.arange(C)

	#Sets the leading party to party number 0 in each constituency as a default
	leadingParty = np.zeros(C)

	#The whole loop is initialization
	for c in range(C):
		#Number of parties in constituency c
		K = len(listParties[c])

		#Index of the constiuency in the undecided constituency list
		indexC = np.where(undecidedConstituencies == c)

		#Initialization
		for _ in range(init_batch):
			#Calculates the fraction of unseen votes won by each candidate for constiuency c
			norm = [float(i)/sum(unseenVotes[c]) for i in unseenVotes[c]]
			#Samples with probabilities calculated above, returns a one hot vector
			# print(constituencies[c], unseenVotes[c], norm)
			vote = np.random.multinomial(1, norm)
			#Updates the seen and unseen votes accordingly
			unseenVotes[c] -= vote
			seenVotes[c] += vote

		#Updates the lower and upper bounds for each party in constituency c
		for k in range(K):
			
			tempL, tempU = binBounds(alpha/(K*C), N0[c], a, b, sum(seenVotes[c]), seenVotes[c][k])

			Nl[c][k] = max(Nl[c][k], tempL)
			Nu[c][k] = min(Nu[c][k], tempU)

		
		#Party with the most votes currently in constituency c
		constiWinner = np.argmax(seenVotes[c])

		#Differnce between lower bound of current constituency winner and the greatest of the upper bounds of the remaining
		constiTerm = Nl[c][constiWinner] - max([x for i,x in enumerate(Nu[c]) if i!=constiWinner])

		#Sets the leading party of constituency c to the current winner
		leadingParty[indexC] = Parties.index(listParties[c][constiWinner])

		#Constituency is decided if the lower bound of current winner is above the upper bound of all other parties
		if constiTerm > 0:

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
				Cu[p] = seenWins[p]

				for ci in undecidedConstituencies:
					if p in listPartyIDs[ci]:
						pIndex = listPartyIDs[ci].index(p)
						#party may win a constiuency if its upper confidence bound is greater than the
						#max of all lower confidence bounds
						if Nu[ci][pIndex] >= max(Nl[ci]):
							Cu[p] += 1

			#winner of the election is the party with the greatest lower bound in number of constituencies won  
			winner = np.argmax(Cl)

			#terminate if the lower bound of the leading party is greater than the upper bound of any other party
			term = Cl[winner] - max(Cu[np.arange(len(Cu)) != winner])

			print_data = {}
			print_data['constituency_id'] = str(c)
			print_data['constituency'] = str(constituencies[c])
			print_data['winning_party'] = str(Parties[winPartyID])
			print_data['consti_votes'] = str(sum(seenVotes[c]))
			print_data['total_votes'] = str(sum(map(sum,seenVotes)))
			print_data['undecided_constituencies'] = str(N)
			# for i,vote in enumerate(listVotes[c]):
				# print_data['constituency_votes_{}'.format(Parties[i])] = str(vote)
			# for p in range(P):
				# print_data['lower_bound_{}'.format(p)] = str(Cl[p])
				# print_data['upper_bound_{}'.format(p)] = str(Cu[p])
			print(print_data)
			f.write(json.dumps(print_data) + '\n')
			#Every time 5 constituencies are decided print the following:
			#Number of undecided constituencies, current constiuency, winner of current constituency, 
			#Total votes seen across all constituencies, total votes seen in constituency c,
			#lower confidence bound of current overall winner, greatest among upperbounds of other parties
			if N % 5 == 0 and False:

				print('Number of Undecided Constituencies: {}'.format(N), 'Constituency Decided: {}'.format(constituencies[c]), 
				'Winning Party: {}'.format(Parties[winPartyID]), 'Total votes sampled: {}'.format(sum(map(sum, seenVotes))), 
				'Votes sampled in this constituency: {}'.format(sum(seenVotes[c])))
				print('Lower bound of the overall winning party: {}'.format(Cl[winner]), 'Max among upper bounds of remaining parties: {}'
				.format(max(Cu[np.arange(len(Cu)) != winner])))
			
			#if terminating, print the number of constituencies decided, winning party,
			#constituencies won by the winner, total votes counted		  
			if term > 0:

				totalVotesCounted = sum(map(sum, seenVotes))
				print("Decided constituencies = ", sum(seenWins))
				print(Parties[winner], ", Seats won = ", seenWins[winner], ", Total votes counted = ", totalVotesCounted)
				f.close()
				return sum(seenWins), winner, totalVotesCounted, seenVotes, listVotes

	counter = 0
	#The main part of the algorithm
	while N > 0:
		#list of number of undecided leads for each party
		countWinning = np.array([np.count_nonzero(leadingParty == p) for p in range(P)])

		#update the upper and lower bounds for each party
		for p in range(P):

			Cl[p] = seenWins[p]
			Cu[p] = seenWins[p]

			for ci in undecidedConstituencies:
				if p in listPartyIDs[ci]:
					pIndex = listPartyIDs[ci].index(p)
					#party may win a constituency if its upper bound is greater than
					#the maximum of all lower bounds
					if Nu[ci][pIndex] >= max(Nl[ci]):
						Cu[p] += 1
			

##        pb, pa = np.argsort(countWinning + seenWins)[-2:]

		#The winning party including both leads and decided constituencies
		pa = np.argmax(countWinning + seenWins)

		
		idx = np.arange(P)
		a1 = np.delete(idx, pa)
		#Index of party with highest upper confidence bound amongst the remaining
		a2 = np.argmax(np.delete(Cu, pa))
		
		#Party with highest upper confidence bound amongst the remaining
		pb = a1[a2]
		
		#Array of zeros, size = number of constituencies
		aUCB = np.zeros(C)

		#calculates aUCB of each constituency as the fraction of votes (upper bounded) won by the currently winning party
		for c in undecidedConstituencies:

			if pa in listPartyIDs[c]:

				paIndex = listPartyIDs[c].index(pa)
				
				aUCB[c] = Nu[c][paIndex] / N0[c]


		#If the party has not won any votes (upper bounded) in the remaining constituencies, choose one at random
		if max(aUCB) == 0:
			print("Random A")
			c = randChoice(undecidedConstituencies)
		#Otherwise choose the constituency with highest aUCB value
		else:
			c = np.argmax(aUCB)
		
		#The index of the chosen constituency in the undecided constituencies list
		indexC = np.where(undecidedConstituencies == c)

		#Number of parties contesting constituency c
		K = len(listParties[c])

		counter += 1
		#Uniformly sample a batch of voters from chosen constituency
		for _ in range(batch):

			norm = [float(i)/sum(unseenVotes[c]) for i in unseenVotes[c]]
			# print(constituencies[c], unseenVotes[c], norm)
			vote = np.random.multinomial(1, norm)
			
			unseenVotes[c] -= vote
			seenVotes[c] += vote

		#update lower and upper bounds for each party in constituency
		for k in range(K):
			
			tempL, tempU = binBounds(alpha/(K*C), N0[c], a, b, sum(seenVotes[c]), seenVotes[c][k])

			Nl[c][k] = max(Nl[c][k], tempL)
			Nu[c][k] = min(Nu[c][k], tempU)


		constiWinner = np.argmax(seenVotes[c])

		constiTerm = Nl[c][constiWinner] - max([x for i,x in enumerate(Nu[c]) if i!=constiWinner])

		

		leadingParty[indexC] = Parties.index(listParties[c][constiWinner])

		if constiTerm > 0:

			N = N - 1
		 
			winPartyID = int(leadingParty[indexC])
			
			leadingParty = np.delete(leadingParty, indexC)
			undecidedConstituencies = np.delete(undecidedConstituencies, indexC)

			seenWins[winPartyID] += 1

			for p in range(P):

				Cl[p] = seenWins[p]
				Cu[p] = seenWins[p]

				for ci in undecidedConstituencies:
					if p in listPartyIDs[ci]:
						pIndex = listPartyIDs[ci].index(p)
						if Nu[ci][pIndex] >= max(Nl[ci]):
							Cu[p] += 1

			winner = np.argmax(Cl)

			term = Cl[winner] - max(Cu[np.arange(len(Cu)) != winner])

			print_data = {}
			print_data['constituency_id'] = str(c)
			print_data['constituency'] = str(constituencies[c])
			print_data['winning_party'] = str(Parties[winPartyID])
			print_data['consti_votes'] = str(sum(seenVotes[c]))
			print_data['total_votes'] = str(sum(map(sum,seenVotes)))
			print_data['undecided_constituencies'] = str(N)
			print_data['Party A'] = Parties[pa]
			print_data['Party B'] = Parties[pb]
			print_data['UCB of A'] = Cu[pa]
			print_data['LCB of A'] = Cl[pa]
			print_data['UCB of B'] = Cu[pb]
			print_data['LCB of B'] = Cl[pb]

			# for party, vote in zip(listParties[c], listVotes[c]):
				# print_data['constituency_votes_{}'.format(party)] = str(vote)
			# for p in range(P):
				# print_data['leads_{}'.format(p)] = str(countWinning[p])
				# print_data['lower_bound_{}'.format(p)] = str(Cl[p])
				# print_data['upper_bound_{}'.format(p)] = str(Cu[p])
			print(print_data)
			f.write(json.dumps(print_data) + '\n')
			
			if N % 5 == 0 and False:

##                print("*")
##                print(np.argsort(seenWins + countWinning)[-4:], sum(seenWins) + len(leadingParty))
##                print("**")
##                print(Cu[np.argsort(seenWins + countWinning)[-4:]], Cl[np.argsort(seenWins + countWinning)[-4:]])
##                print("***")
##                print(seenWins[np.argsort(seenWins + countWinning)[-4:]], countWinning[np.argsort(seenWins + countWinning)[-4:]])
##

				print('Leading party overall (before this constituency): {}'.format(pa),
				'Party with highest UCB among the remaining (before this constituency): {}'.format(pb))
				print('Remaining Constituencies: {}'.format(N), 'Constituency Decided: {}'.format(constituencies[c]), 
				'Winning party: {}'.format(Parties[winPartyID]), 'Total votes seen: {}'.format(sum(map(sum, seenVotes))), 
				'Votes seen in this constituency: {}'.format(sum(seenVotes[c])))
				print('Highest lower bound: {}'.format(Cl[winner]), 'Highest upper bound among remaining: {}'
				.format(max(Cu[np.arange(len(Cu)) != winner])))
					  
			if term > 0:

				totalVotesCounted = sum(map(sum, seenVotes))
				print("Decided constituencies = ", sum(seenWins))
				print('Winner of election: {}'.format(Parties[winner]), ", Seats won = ", seenWins[winner], ", Total votes counted = ", totalVotesCounted)

				f.close()
				return sum(seenWins), winner, totalVotesCounted, seenVotes, listVotes


##        countWinning = np.array([np.count_nonzero(np.array(leadingParty) == p) for p in range(P)])

		bLCB = np.ones(C)

		#Calculates bLCB as the number of votes (lower bounded) won by the second party in each constituency
		for c in undecidedConstituencies:

			if pb in listPartyIDs[c]:
				pbIndex = listPartyIDs[c].index(pb)
				if Nu[c][pbIndex] >= max(Nl[c]):
					bLCB[c] = Nl[c][pbIndex] / N0[c]

		#In the (rare) event that party b has won all votes in the undecided constituencies choose at random
		if min(bLCB) == 1:
			print("Random B")
			c = randChoice(undecidedConstituencies)
			
		else:
			c = np.argmin(bLCB)

		indexC = np.where(undecidedConstituencies == c)

		K = len(listParties[c])
		
		counter += 1


		for _ in range(batch):

			norm = [float(i)/sum(unseenVotes[c]) for i in unseenVotes[c]]
			# print(constituencies[c], unseenVotes[c], norm)
			vote = np.random.multinomial(1, norm)
			unseenVotes[c] -= vote
			seenVotes[c] += vote

		for k in range(K):
			
			tempL, tempU = binBounds(alpha/(K*C), N0[c], a, b, sum(seenVotes[c]), seenVotes[c][k])

			Nl[c][k] = max(Nl[c][k], tempL)
			Nu[c][k] = min(Nu[c][k], tempU)

		constiWinner = np.argmax(seenVotes[c])

		constiTerm = Nl[c][constiWinner] - max([x for i,x in enumerate(Nu[c]) if i!=constiWinner])

		leadingParty[indexC] = Parties.index(listParties[c][constiWinner])

		if constiTerm > 0:

			N = N - 1
			
			winPartyID = int(leadingParty[indexC])
			
			leadingParty = np.delete(leadingParty, indexC)
			undecidedConstituencies = np.delete(undecidedConstituencies, indexC)

			seenWins[winPartyID] += 1

			for p in range(P):

				Cl[p] = seenWins[p]
				Cu[p] = seenWins[p]

				for ci in undecidedConstituencies:
					if p in listPartyIDs[ci]:
						pIndex = listPartyIDs[ci].index(p)
						if Nu[ci][pIndex] >= max(Nl[ci]):
							Cu[p] += 1

			winner = np.argmax(Cl)

			term = Cl[winner] - max(Cu[np.arange(len(Cu)) != winner])

			print_data = {}
			print_data['constituency_id'] = str(c)
			print_data['constituency'] = str(constituencies[c])
			print_data['winning_party'] = str(Parties[winPartyID])
			print_data['consti_votes'] = str(sum(seenVotes[c]))
			print_data['total_votes'] = str(sum(map(sum,seenVotes)))
			print_data['Party A'] = Parties[pa]
			print_data['Party B'] = Parties[pb]
			print_data['UCB of A'] = Cu[pa]
			print_data['LCB of A'] = Cl[pa]
			print_data['UCB of B'] = Cu[pb]
			print_data['LCB of B'] = Cl[pb]

			# print_data['undecided_constituencies'] = str(N)
			# for party, vote in zip(listParties[c], listVotes[c]):
				# print_data['constituency_votes_{}'.format(party)] = str(vote)
			# for p in range(P):
				# print_data['leads_{}'.format(p)] = str(countWinning[p])
				# print_data['lower_bound_{}'.format(p)] = str(Cl[p])
				# print_data['upper_bound_{}'.format(p)] = str(Cu[p])
			print(print_data)
			f.write(json.dumps(print_data) + '\n')

			if N % 5 == 0 and False:

##                print("*")
##                print(np.argsort(seenWins + countWinning)[-4:], sum(seenWins) + len(leadingParty))
##                print("**")
##                print(Cu[np.argsort(seenWins + countWinning)[-4:]], Cl[np.argsort(seenWins + countWinning)[-4:]])
##                print("***")
##                print(seenWins[np.argsort(seenWins + countWinning)[-4:]], countWinning[np.argsort(seenWins + countWinning)[-4:]])

				print('Leading party overall (before this constituency): {}'.format(pa),
				'Party with highest UCB among the remaining (before this constituency): {}'.format(pb))
				print('Remaining Constituencies: {}'.format(N), 'Constituency Decided: {}'.format(constituencies[c]), 
				'Winning party: {}'.format(Parties[winPartyID]), 'Total votes seen: {}'.format(sum(map(sum, seenVotes))), 
				'Votes seen in this constituency: {}'.format(sum(seenVotes[c])))
				print('Highest lower bound: {}'.format(Cl[winner]), 'Highest upper bound among remaining: {}'
				.format(max(Cu[np.arange(len(Cu)) != winner])))
					  
			if term > 0:

				totalVotesCounted = sum(map(sum, seenVotes))
				print("Decided constituencies = ", sum(seenWins))
				print('Winner of election: {}'.format(Parties[winner]), ", Seats won = ", seenWins[winner], ", Total votes counted = ", totalVotesCounted)
				
				f.close()
				return sum(seenWins), winner, totalVotesCounted, seenVotes, listVotes

if __name__ == "__main__":


	alpha = 10**-1
	batch = 100
	init_batch = 100
	data = "India2014"
	
	stoppingC, winner, stoppingVotes, seenVotes = runBanditElectionLUCB(data, alpha, batch, init_batch)

	print(stoppingC, winner, stoppingVotes)
