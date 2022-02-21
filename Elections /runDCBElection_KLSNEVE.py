import numpy as np
import pandas as pd
from math import e
from util_ppr2 import *
import json


def randChoice(x):
	
	return np.random.choice(x)

## LUCB inspired algorithm

global_delta = -1

def eval_func(delta_val):
	return 2*(e**2) * delta_val*np.exp(-1*delta_val)


def solve_equation(delta):
	left = 0
	right = 100
	mid = int((left+right)/2)
	itera = 0
	while abs( eval_func(mid) - delta) > 1e-11:

		if(abs(right - left)< 1e-10):
			break
		if(delta<eval_func(mid)):
			left = mid
		else:
			right = mid
		mid = (left+right)/2
		itera += 1

	return mid


def exploration_rate(time,new_delta):

	rate = new_delta*(1+np.log(new_delta))*np.log(np.log(time))/((new_delta-1)*np.log(new_delta)) + new_delta

	return rate


def get_kl_lower_bound(p,beta, time):
		lo = 0
		hi = p
		q = (lo+hi)/2
		lhs = KL_div(p,q)*time

		while abs(beta-lhs) > 1e-5:
			if abs(hi-lo) < 1e-7:
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

		while abs(beta-lhs) > 1e-5:
			if abs(hi-lo) < 1e-7:
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

			
def runDCBElection_KLSNEVE(data, alpha, tracefile, batch = 1, init_batch = 1, a = 1, b = 1):
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
	Cu = np.ones(P) * C

	# labels of the people accounted
	labels = [[set() for _ in inner] for inner in listVotes]
	votesLabelled = 0

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
			seenVotes[c] += vote

			# updating the labelled information
			party_index = np.argmax(vote)
			person_label = np.random.randint(0, unseenVotes[c][party_index])
			if person_label not in labels[c][party_index]:
				labels[c][party_index].add(person_label)
				votesLabelled += 1
		
		#Party with the most votes currently in constituency c
		constiWinner = np.argmax(seenVotes[c])

		#Difference between lower bound of current constituency winner and the greatest of the upper bounds of the remaining
		winner_decided = True
		num_parties = len(seenVotes[c])
		new_delta = solve_equation(alpha / ((num_parties - 1) * C))
		for k in range(len(seenVotes[c])):
			if (k == constiWinner): continue
			mean_w = float((seenVotes[c][constiWinner] * 1.0) / (seenVotes[c][constiWinner] + seenVotes[c][k]))
			mean_k = 1.0 - mean_w
			beta = exploration_rate(seenVotes[c][constiWinner] + seenVotes[c][k], new_delta)
			w_l = get_kl_lower_bound(mean_w, beta, seenVotes[c][constiWinner] + seenVotes[c][k])
			k_h = get_kl_upper_bound(mean_k, beta, seenVotes[c][constiWinner] + seenVotes[c][k])
			winner_decided = winner_decided and (w_l > k_h)
			if not winner_decided:
				break


		#Sets the leading party of constituency c to the current winner
		leadingParty[indexC] = Parties.index(listParties[c][constiWinner])

		#Constituency is decided if the lower bound of current winner is above the upper bound of all other parties
		if winner_decided:

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
						if seenVotes[ci][pIndex] == max(seenVotes[ci]):
							Cu[p] += 1
						else:
							differentiated = False
							num_parties = len(seenVotes[ci])
							new_delta = solve_equation(alpha / ((num_parties - 1) * C))
							for k in range(num_parties):
								if k == pIndex: continue
								mean_p = float((seenVotes[ci][pIndex] * 1.0) / (seenVotes[ci][pIndex] + seenVotes[ci][k]))
								mean_k = 1.0 - mean_p
								beta = exploration_rate(seenVotes[ci][pIndex] + seenVotes[ci][k], new_delta)
								p_h = get_kl_upper_bound(mean_p, beta, seenVotes[ci][pIndex] + seenVotes[ci][k])
								k_l = get_kl_lower_bound(mean_k, beta, seenVotes[ci][pIndex] + seenVotes[ci][k])
								if (p_h < k_l):
									differentiated = True
									break
							if not differentiated:
								Cu[p] += 1


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
				print_data['votesLabelled'] = str(votesLabelled)
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
			Cu[p] = seenWins[p]

			for ci in undecidedConstituencies:
				if p in listPartyIDs[ci]:
					pIndex = listPartyIDs[ci].index(p)
					#party may win a constituency if its upper bound is greater than
					#the maximum of all lower bounds
					if seenVotes[ci][pIndex] == max(seenVotes[ci]):
						Cu[p] += 1
					else:
						differentiated = False
						num_parties = len(seenVotes[ci])
						new_delta = solve_equation(alpha / ((num_parties - 1) * C))
						for k in range(num_parties):
							if k == pIndex: continue
							mean_p = float((seenVotes[ci][pIndex] * 1.0) / (seenVotes[ci][pIndex] + seenVotes[ci][k]))
							mean_k = 1.0 - mean_p
							beta = exploration_rate(seenVotes[ci][pIndex] + seenVotes[ci][k], new_delta)
							p_h = get_kl_upper_bound(mean_p, beta, seenVotes[ci][pIndex] + seenVotes[ci][k])
							k_l = get_kl_lower_bound(mean_k, beta, seenVotes[ci][pIndex] + seenVotes[ci][k])
							if (p_h < k_l):
								differentiated = True
								break
						if not differentiated:
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
		hLCB = np.zeros(C)
		diff_a = -1e18*np.ones(C)
		#calculates aUCB of each constituency as the fraction of votes (upper bounded) won by the currently winning party
		for c in undecidedConstituencies:

			if pa in listPartyIDs[c]:

				paIndex = listPartyIDs[c].index(pa)
				
				min_diff = 1e18
				num_parties = len(seenVotes[c])
				new_delta = solve_equation(alpha / ((num_parties - 1) * C))
				for k in range(len(seenVotes[c])):
					if k == paIndex: continue
					mean_w = seenVotes[c][paIndex] / (seenVotes[c][paIndex] + seenVotes[c][k])
					mean_k = 1 - mean_w
					beta = exploration_rate(seenVotes[c][paIndex] + seenVotes[c][k], new_delta)
					w_h = get_kl_upper_bound(mean_w, beta, seenVotes[c][paIndex] + seenVotes[c][k])
					k_l = get_kl_lower_bound(mean_k, beta, seenVotes[c][paIndex] + seenVotes[c][k])
					min_diff = min(min_diff, w_h - k_l)
				diff_a[c] = min_diff

		#If the party has not won any votes (upper bounded) in the remaining constituencies, choose one at random
		if max(diff_a) == -1e18 :
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
		
		#Uniformly sample a batch of voters from chosen constituency
		for _ in range(batch):

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

		constiWinner = np.argmax(seenVotes[c])
		winner_decided = True
		num_parties = len(seenVotes[c])
		new_delta = solve_equation(alpha / ((num_parties - 1) * C))
		for k in range(len(seenVotes[c])):
			if (k == constiWinner): continue
			mean_w = float((seenVotes[c][constiWinner] * 1.0) / (seenVotes[c][constiWinner] + seenVotes[c][k]))
			mean_k = 1.0 - mean_w
			beta = exploration_rate(seenVotes[c][constiWinner] + seenVotes[c][k], new_delta)
			w_l = get_kl_lower_bound(mean_w, beta, seenVotes[c][constiWinner] + seenVotes[c][k])
			k_h = get_kl_upper_bound(mean_k, beta, seenVotes[c][constiWinner] + seenVotes[c][k])
			winner_decided = winner_decided and (w_l > k_h)
			if not winner_decided:
				break

		leadingParty[indexC] = Parties.index(listParties[c][constiWinner])

		if winner_decided:

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
						if seenVotes[ci][pIndex] == max(seenVotes[ci]):
							Cu[p] += 1
						else:
							differentiated = False
							num_parties = len(seenVotes[ci])
							new_delta = solve_equation(alpha / ((num_parties - 1) * C))
							for k in range(num_parties):
								if k == pIndex: continue
								mean_p = float((seenVotes[ci][pIndex] * 1.0) / (seenVotes[ci][pIndex] + seenVotes[ci][k]))
								mean_k = 1.0 - mean_p
								beta = exploration_rate(seenVotes[ci][pIndex] + seenVotes[ci][k], new_delta)
								p_h = get_kl_upper_bound(mean_p, beta, seenVotes[ci][pIndex] + seenVotes[ci][k])
								k_l = get_kl_lower_bound(mean_k, beta, seenVotes[ci][pIndex] + seenVotes[ci][k])
								if (p_h < k_l):
									differentiated = True
									break
							if not differentiated:
								Cu[p] += 1

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
				print_data['votesLabelled'] = str(votesLabelled)
				print('Election Complete')
				print(print_data)
				f.write(json.dumps(print_data) + '\n')

				f.close()
				return sum(seenWins), winner, totalVotesCounted, seenVotes, votesLabelled

		bLCB = np.ones(C)

		hUCB = np.zeros(C)
		diff_b = -1e18*np.ones(C)
		#Calculates bLCB as the number of votes (lower bounded) won by the second party in each constituency
		for c in undecidedConstituencies:

			if pb in listPartyIDs[c]:
				pbIndex = listPartyIDs[c].index(pb)

				max_diff = -1e18
				num_parties = len(seenVotes[c])
				new_delta = solve_equation(alpha / ((num_parties - 1) * C))
				for k in range(num_parties):
					if pbIndex == k: continue
					mean_w = float(seenVotes[c][pbIndex] * 1.0 / (seenVotes[c][pbIndex] + seenVotes[c][k]))
					mean_k = 1 - mean_w
					beta = exploration_rate(seenVotes[c][pbIndex] + seenVotes[c][k], new_delta)
					w_h = get_kl_upper_bound(mean_w, beta, seenVotes[c][pbIndex] + seenVotes[c][k])
					k_l = get_kl_lower_bound(mean_k, beta, seenVotes[c][pbIndex] + seenVotes[c][k])
					if (w_h < k_l):
						max_diff = -1e18
						break
					w_l = get_kl_lower_bound(mean_w, beta, seenVotes[c][pbIndex] + seenVotes[c][k])
					k_h = get_kl_upper_bound(mean_k, beta, seenVotes[c][pbIndex] + seenVotes[c][k])
					max_diff = max(max_diff, k_h - w_l)

					
				diff_b[c] = max_diff


		if max(diff_b) == -1e18:
			print("Random B")
			c = randChoice(undecidedConstituencies)
			
		else:
			c = np.argmax(diff_b)

		indexC = np.where(undecidedConstituencies == c)

		K = len(listParties[c])
		
		counter += 1


		for _ in range(batch):

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

		constiWinner = np.argmax(seenVotes[c])
		winner_decided = True
		num_parties = len(seenVotes[c])
		new_delta = solve_equation(alpha / ((num_parties - 1) * C))
		for k in range(len(seenVotes[c])):
			if (k == constiWinner): continue
			mean_w = float((seenVotes[c][constiWinner] * 1.0) / (seenVotes[c][constiWinner] + seenVotes[c][k]))
			mean_k = 1.0 - mean_w
			beta = exploration_rate(seenVotes[c][constiWinner] + seenVotes[c][k], new_delta)
			w_l = get_kl_lower_bound(mean_w, beta, seenVotes[c][constiWinner] + seenVotes[c][k])
			k_h = get_kl_upper_bound(mean_k, beta, seenVotes[c][constiWinner] + seenVotes[c][k])
			winner_decided = winner_decided and (w_l > k_h)
			if not winner_decided:
				break
		
		leadingParty[indexC] = Parties.index(listParties[c][constiWinner])

		if winner_decided:

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
						if seenVotes[ci][pIndex] == max(seenVotes[ci]):
							Cu[p] += 1
						else:
							differentiated = False
							num_parties = len(seenVotes[ci])
							new_delta = solve_equation(alpha / ((num_parties - 1) * C))
							for k in range(num_parties):
								if k == pIndex: continue
								mean_p = float((seenVotes[ci][pIndex] * 1.0) / (seenVotes[ci][pIndex] + seenVotes[ci][k]))
								mean_k = 1.0 - mean_p
								beta = exploration_rate(seenVotes[ci][pIndex] + seenVotes[ci][k], new_delta)
								p_h = get_kl_upper_bound(mean_p, beta, seenVotes[ci][pIndex] + seenVotes[ci][k])
								k_l = get_kl_lower_bound(mean_k, beta, seenVotes[ci][pIndex] + seenVotes[ci][k])
								if (p_h < k_l):
									differentiated = True
									break
							if not differentiated:
								Cu[p] += 1

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
				print_data['votesLabelled'] = str(votesLabelled)
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
