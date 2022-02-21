import numpy as np
import pandas as pd
from util_ppr2 import *
import json


def randChoice(x):
	
	return np.random.choice(x)

## LUCB inspired algorithm
			
def runDCBElection_PPR2(data, alpha, tracefile, batch = 1, init_batch = 1, a = 1, b = 1):
	f = open(tracefile, 'w')
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

		#List of votes per party for the constituency c
		try:
			table['Votes'] = table['Votes'].str.replace(',','')
		except:
			pass
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
	Cu = np.zeros(P)
	countContested = np.zeros(P)

	# labels of the people accounted
	labels = [[set() for _ in inner] for inner in listVotes]
	votesLabelled = 0

	#Sets Cu[i] to the number of constituencies contested by party i - better upper bound
	for p in range(P):
		countContested[p] = sum(row.count(Parties[p]) for row in listParties)
		Cu[p] = countContested[p]

	#The total number of votes (population size) for each constituency
	N0 = [sum(inner) for inner in listVotes]
	
	#This stores everything we need to sample according to the rule for the party A in DCB algo
	Aval = [[0] * len(inner) for inner in listVotes]

	#This stores everything we need to sample according to the rule for the party B in DCB algo
	Bval = [[0] * len(inner) for inner in listVotes]
	hasLost = [[False] * len(inner) for inner in listVotes]
	#Initializes the number of wins to 0 for each party
	seenWins = np.zeros(P)
	seenLosses = np.zeros(P)
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
		constiTerm = False
		for _ in range(init_batch):
			if sum(unseenVotes[c]) == 0:
				constiTerm = True
				break
			#Calculates the fraction of unseen votes won by each candidate for constiuency c
			norm = [float(i)/sum(unseenVotes[c]) for i in unseenVotes[c]]
			#Samples with probabilities calculated above, returns a one hot vector
			# print(constituencies[c], unseenVotes[c], norm)
			vote = np.random.multinomial(1, norm)
			#Updates the seen and unseen votes accordingly
			seenVotes[c] += vote

			# updating the labelled information
			party_index = np.argmax(vote)
			person_label = np.random.randint(0, unseenVotes[c][party_index])
			if person_label not in labels[c][party_index]:
				labels[c][party_index].add(person_label)
				votesLabelled += 1

		#Updates the lower and upper bounds for each party in constituency c
		for k in range(K):
			if sum(unseenVotes[c]) == 0:
				break
			if hasLost[c][k]:
				continue
			#Get A val
			#Two cases - i) party k is first, ii) party k is not first
			if k == np.argmax(seenVotes[c]):
				second = np.argsort(seenVotes[c])[-2]
				lcb, ucb = binBounds2(alpha/((K - 1)*C), a, b, seenVotes[c][k] + seenVotes[c][second], seenVotes[c][k])
				#If the first party is separated from the second we can terminate
				if lcb > 0.5:
					constiTerm = True
			else:
				first = np.argmax(seenVotes[c])
				lcb, ucb = binBounds2(alpha/((K - 1)*C), a, b, seenVotes[c][k] + seenVotes[c][first], seenVotes[c][k])
			#Aval[c][k] is the difference between the ucb of k and max_i(lcb_i) such that i != k	
			Aval[c][k] = 2*ucb - 1
			#Get B val
			#Two cases - i) party k is first, ii) party k is not first
			if k == np.argmax(seenVotes[c]):
				second = np.argsort(seenVotes[c])[-2]
				lcb, ucb = binBounds2(alpha/((K - 1)*C), a, b, seenVotes[c][k] + seenVotes[c][second], seenVotes[c][second])
			else:
				first = np.argmax(seenVotes[c])
				lcb, ucb = binBounds2(alpha/((K - 1)*C), a, b, seenVotes[c][k] + seenVotes[c][first], seenVotes[c][first])
				partyID = listPartyIDs[c][k]
				#we can say a party has lost if it separated from any other party
				if lcb > 0.5:
					hasLost[c][k] = True
					seenLosses[partyID] += 1
			#Bval[c][k] is the difference between max_i(ucb_i) such that i != k and lcb of k	
			Bval[c][k] = 2*ucb - 1

		
		#Party with the most votes currently in constituency c
		constiWinner = np.argmax(seenVotes[c])
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

			#terminate if the lower bound of the leading party is greater than the upper bound of any other party
			term = Cl[winner] - max(Cu[np.arange(len(Cu)) != winner])

			print_data = {}
			print_data['constituency_id'] = str(c)
			print_data['constituency_name'] = str(constituencies[c])
			print_data['consti_winner'] = str(Parties[winPartyID])
			print_data['consti_votes'] = str(sum(seenVotes[c]))
			print_data['total_votes'] = str(sum(map(sum,seenVotes)))
			print_data['undecided_constituencies'] = str(N)
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

	counter = 0
	#The main part of the algorithm
	while N > 0:
		#list of number of undecided leads for each party
		countWinning = np.array([np.count_nonzero(leadingParty == p) for p in range(P)])

		#update the upper and lower bounds for each party
		for p in range(P):
			Cl[p] = seenWins[p]
			Cu[p] = countContested[p] - seenLosses[p]


		#The winning party including both leads and decided constituencies
		pa = np.argmax(countWinning + seenWins)

		idx = np.arange(P)
		a1 = np.delete(idx, pa)
		#Index of party with highest upper confidence bound amongst the remaining
		a2 = np.argmax(np.delete(Cu, pa))
		#Party with highest upper confidence bound amongst the remaining
		pb = a1[a2]
		
		#Array of zeros, size = number of constituencies
		diff_a = -1*np.ones(C)
		
		for c in undecidedConstituencies:
			if pa in listPartyIDs[c]:
				paIndex = listPartyIDs[c].index(pa)
				if hasLost[c][paIndex]:
					continue
				diff_a[c] = Aval[c][paIndex]

		#If the party has not won any votes (upper bounded) in the remaining constituencies, choose one at random
		if max(diff_a) == -1 :
			print("Random A")
			c = randChoice(undecidedConstituencies)
		#Otherwise choose the constituency with highest aUCB value
		else:
			c = np.argmax(diff_a)
		#The index of the chosen constituency in the undecided constituencies list
		indexC = np.where(undecidedConstituencies == c)

		#Number of parties contesting constituency c
		K = len(listParties[c])

		counter += 1
		
		constiTerm = False
		#Uniformly sample a batch of voters from chosen constituency
		for _ in range(batch):
			# print(unseenVotes[c])
			if sum(unseenVotes[c]) == 0:
				constiTerm = True
				first = np.argmax(seenVotes[c])
				for k in range(len(seenVotes[c])):
					if k == first or hasLost[c][k]:
						continue
					hasLost[c][k] = True
					partyIndex = listPartyIDs[c][k]
					seenLosses[partyIndex] += 1

				break
			norm = [float(i)/sum(unseenVotes[c]) for i in unseenVotes[c]]
			# print(constituencies[c], unseenVotes[c], norm)
			vote = np.random.multinomial(1, norm)
			
			# unseenVotes[c] -= vote
			seenVotes[c] += vote

			# updating the labelled information
			party_index = np.argmax(vote)
			person_label = np.random.randint(0, unseenVotes[c][party_index])
			if person_label not in labels[c][party_index]:
				labels[c][party_index].add(person_label)
				votesLabelled += 1

		for k in range(K):
			if sum(unseenVotes[c]) == 0:
				break
			
			if hasLost[c][k]:
				continue
			#Get A val
			#Two cases - i) party k is first, ii) party k is not first
			if k == np.argmax(seenVotes[c]):
				second = np.argsort(seenVotes[c])[-2]
				lcb, ucb = binBounds2(alpha/((K - 1)*C), a, b, seenVotes[c][k] + seenVotes[c][second], seenVotes[c][k])
				if lcb > 0.5:
					constiTerm = True
				firstVotes = seenVotes[c][k]
				secondVotes = seenVotes[c][second]
				remainingVotes = sum(unseenVotes[c])
				if secondVotes + remainingVotes < firstVotes:
					constiTerm = True
			else:
				first = np.argmax(seenVotes[c])
				lcb, ucb = binBounds2(alpha/((K - 1)*C), a, b, seenVotes[c][k] + seenVotes[c][first], seenVotes[c][k])
			#Aval[c][k] is the difference between the ucb of k and max_i(lcb_i) such that i != k	
			Aval[c][k] = 2*ucb - 1
			#Get B val
			#Two cases - i) party k is first, ii) party k is not first
			if k == np.argmax(seenVotes[c]):
				second = np.argsort(seenVotes[c])[-2]
				lcb, ucb = binBounds2(alpha/((K - 1)*C), a, b, seenVotes[c][k] + seenVotes[c][second], seenVotes[c][second])
			else:
				first = np.argmax(seenVotes[c])
				lcb, ucb = binBounds2(alpha/((K - 1)*C), a, b, seenVotes[c][k] + seenVotes[c][first], seenVotes[c][first])
				partyID = listPartyIDs[c][k]
				#we can say a party has lost if it separated from any other party
				if lcb > 0.5:
					hasLost[c][k] = True
					seenLosses[partyID] += 1
			#Bval[c][k] is the difference between max_i(ucb_i) such that i != k and lcb of k	
			Bval[c][k] = 2*ucb - 1


		constiWinner = np.argmax(seenVotes[c])
		secondPlace = np.argsort(seenVotes[c])[-2]		

		leadingParty[indexC] = Parties.index(listParties[c][constiWinner])

		if constiTerm:

			N = N - 1
		 
			winPartyID = int(leadingParty[indexC])
			
			leadingParty = np.delete(leadingParty, indexC)
			undecidedConstituencies = np.delete(undecidedConstituencies, indexC)

			seenWins[winPartyID] += 1

			for p in range(P):
				Cl[p] = seenWins[p]
				Cu[p] = countContested[p] - seenLosses[p]

			winner = np.argmax(Cl)

			term = Cl[winner] - max(Cu[np.arange(len(Cu)) != winner])

			print_data = {}
			print_data['constituency_id'] = str(c)
			print_data['constituency'] = str(constituencies[c])
			print_data['consti_winner'] = str(Parties[winPartyID])
			print_data['consti_votes'] = str(sum(seenVotes[c]))
			print_data['total_votes'] = str(sum(map(sum,seenVotes)))
			print_data['undecided_constituencies'] = str(N)
			print_data['Party A'] = Parties[pa]
			print_data['Party B'] = Parties[pb]
			print_data['UCB of A'] = str(Cu[pa])
			print_data['LCB of A'] = str(Cl[pa])
			print_data['UCB of B'] = str(Cu[pb])
			print_data['LCB of B'] = str(Cl[pb])
			print(print_data)
			f.write(json.dumps(print_data) + '\n')
			
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

		bLCB = np.ones(C)

		hUCB = np.zeros(C)
		diff_b = -1*np.ones(C)
		#Calculates bLCB as the number of votes (lower bounded) won by the second party in each constituency
		for c in undecidedConstituencies:
			if pb in listPartyIDs[c]:
				pbIndex = listPartyIDs[c].index(pb)
				if hasLost[c][pbIndex]:
					continue
				diff_b[c] = Bval[c][pbIndex]

		#In the (rare) event that party b has won all votes in the undecided constituencies choose at random
		if max(diff_b) == -1:
			print("Random B")
			c = randChoice(undecidedConstituencies)
		else:
			c = np.argmax(diff_b)
		indexC = np.where(undecidedConstituencies == c)

		K = len(listParties[c])
		
		counter += 1

		constiTerm = False
		for _ in range(batch):
			# print(unseenVotes[c])
			if sum(unseenVotes[c]) == 0:
				constiTerm = True
				first = np.argmax(seenVotes[c])
				for k in range(len(seenVotes[c])):
					if k == first or hasLost[c][k]:
						continue
					hasLost[c][k] = True
					partyIndex = listPartyIDs[c][k]
					seenLosses[partyIndex] += 1
				break
			norm = [float(i)/sum(unseenVotes[c]) for i in unseenVotes[c]]
			# print(constituencies[c], unseenVotes[c], norm)
			vote = np.random.multinomial(1, norm)

			# unseenVotes[c] -= vote
			seenVotes[c] += vote

			# updating the labelled information
			party_index = np.argmax(vote)
			person_label = np.random.randint(0, unseenVotes[c][party_index])
			if person_label not in labels[c][party_index]:
				labels[c][party_index].add(person_label)
				votesLabelled += 1

			
		for k in range(K):
			if sum(unseenVotes[c]) == 0:
				break

			if hasLost[c][k]:
				continue
			#Get A val
			#Two cases - i) party k is first, ii) party k is not first
			if k == np.argmax(seenVotes[c]):
				second = np.argsort(seenVotes[c])[-2]
				lcb, ucb = binBounds2(alpha/((K - 1)*C), a, b, seenVotes[c][k] + seenVotes[c][second], seenVotes[c][k])
				if lcb > 0.5:
					constiTerm = True
				firstVotes = seenVotes[c][k]
				secondVotes = seenVotes[c][second]
				remainingVotes = sum(unseenVotes[c])
				if secondVotes + remainingVotes < firstVotes:
					constiTerm = True
			else:
				first = np.argmax(seenVotes[c])
				lcb, ucb = binBounds2(alpha/((K - 1)*C), a, b, seenVotes[c][k] + seenVotes[c][first], seenVotes[c][k])
			#Aval[c][k] is the difference between the ucb of k and max_i(lcb_i) such that i != k	
			Aval[c][k] = 2*ucb - 1
			#Get B val
			#Two cases - i) party k is first, ii) party k is not first
			if k == np.argmax(seenVotes[c]):
				second = np.argsort(seenVotes[c])[-2]
				lcb, ucb = binBounds2(alpha/((K - 1)*C), a, b, seenVotes[c][k] + seenVotes[c][second], seenVotes[c][second])
			else:
				first = np.argmax(seenVotes[c])
				lcb, ucb = binBounds2(alpha/((K - 1)*C), a, b, seenVotes[c][k] + seenVotes[c][first], seenVotes[c][first])
				partyID = listPartyIDs[c][k]
				#we can say a party has lost if it separated from any other party
				if lcb > 0.5:
					hasLost[c][k] = True
					seenLosses[partyID] += 1
			#Bval[c][k] is the difference between max_i(ucb_i) such that i != k and lcb of k	
			Bval[c][k] = 2*ucb - 1

		constiWinner = np.argmax(seenVotes[c])
		
		leadingParty[indexC] = Parties.index(listParties[c][constiWinner])

		if constiTerm:

			N = N - 1
			
			winPartyID = int(leadingParty[indexC])
			
			leadingParty = np.delete(leadingParty, indexC)
			undecidedConstituencies = np.delete(undecidedConstituencies, indexC)

			seenWins[winPartyID] += 1

			for p in range(P):

				Cl[p] = seenWins[p]
				Cu[p] = countContested[p] - seenLosses[p]

			winner = np.argmax(Cl)

			term = Cl[winner] - max(Cu[np.arange(len(Cu)) != winner])

			print_data = {}
			print_data['constituency_id'] = str(c)
			print_data['constituency'] = str(constituencies[c])
			print_data['consti_winner'] = str(Parties[winPartyID])
			print_data['consti_votes'] = str(sum(seenVotes[c]))
			print_data['total_votes'] = str(sum(map(sum,seenVotes)))
			print_data['undecided_constituencies'] = str(N)
			print_data['Party A'] = Parties[pa]
			print_data['Party B'] = Parties[pb]
			print_data['UCB of A'] = str(Cu[pa])
			print_data['LCB of A'] = str(Cl[pa])
			print_data['UCB of B'] = str(Cu[pb])
			print_data['LCB of B'] = str(Cl[pb])
			print(print_data)
			f.write(json.dumps(print_data) + '\n')

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
