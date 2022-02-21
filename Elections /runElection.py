import numpy as np
import pandas as pd
from util import *
from simulate import *
import json

def runElection(data, alpha, tracefile, batch = 1, a = 1, b = 1):
    f = open(tracefile, 'w')

    constituencies = getAllPlaces(f"data/{data}/")

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

        votes = table['Votes'].to_list()
        parties = table['Party'].to_list()
        
        listVotes.append(votes)
        listParties.append(parties)    
    
    indexIndi = 0
    for i, row in enumerate(listParties):
        for j, party in enumerate(row):
            if party == 'Independent':
                listParties[i][j] = f"Independent{indexIndi}"
                indexIndi += 1

                

    Parties = list(set([x for y in listParties for x in y]))
    Parties.sort()

    P = len(Parties)
    
	#List of lists, list[i][j] is the id of party j in the list of parties of the ith constituency
    listPartyIDs = [[Parties.index(k) for k in row] for row in listParties]
            

    try:
        listVotes = [[int(inner.replace(',', ''))for inner in outer] for outer in listVotes]
    except:
        pass

	#Initializes the seen votes to 0 for each candidate in each constituency
    seenVotes = [[0] * len(inner) for inner in listVotes]

    #array of zeros, size = number of constituencies + 1
    N = np.zeros(C+1)
    #array of zeros, size = number of parties x number of constituencies + 1
    Cl = np.zeros((P, C + 1))
    #array with each value equal to number of constituencies, size = number of parties x number of constituencies + 1
    Cu = np.ones((P, C + 1)) * C

    N[0] = C

    #initially all constituencies unseen
    unseenConstituencies = np.arange(C)
    #initially no wins seen for each party
    seenWins = np.zeros(P)

    i = 0
    separated = False
    separated_i = C

    totalVotesCounted = 0

    while N[i] > 0:
        #choose a constituency at random
        c = np.random.choice(unseenConstituencies)
        

        #list of votes and parties for that constituency
        votes = listVotes[c]
        parties = listParties[c]

        #minimum number of votes needed to be checked, the winner at that stage, votes seen at that stage
##        stoppingTimes[c], winners[c] = runExperiment(votes, alpha, True, True, batch = 100)
        stoppingTimes[c], winners[c], seenVotes[c] = runOracleSearch(votes, alpha / (2 * C), batch)
        winners[c] = Parties.index(parties[winners[c]])



        #adds up the total votes counted in each constituency
        totalVotesCounted += stoppingTimes[c]
        #adds a win for the winning party
        seenWins[winners[c]] += 1
        

        #removes the constituency from unseen constituencies
        unseenConstituencies = np.delete(unseenConstituencies, np.where(unseenConstituencies == c))
        #N[i] is number of contituencies remaining
        i = i + 1
        N[i] = N[i-1] - 1

        #Calculate upper and lower bounds at step i
        Cl[:, i] = seenWins
        Cu[:, i] = [(C - (np.sum(seenWins) - seenWins[p])) for p in np.arange(P)]
        # for p in range(P):
            # Cl[p, i], Cu[p, i] = binBounds(alpha/(2 * P), C, a, b, i, seenWins[p])

            ## intersection 
            # Cl[p, i] = max(Cl[p, i-1], Cl[p, i])
            # Cu[p, i] = min(Cu[p, i-1], Cu[p, i])


        winner = np.argmax(Cl[:, i])
        second_place = np.argsort(Cu[:,i])[-2]
        term = Cl[winner, i] - max(Cu[:, i][np.arange(len(Cu[:, i])) != winner])


        print_data = {}
        print_data['constituency_id'] = str(c)
        print_data['constituency_name'] = constituencies[c]
        print_data['consti_winner'] = Parties[winners[c]]
        print_data['consti_votes'] = str(stoppingTimes[c])
        print_data['total_votes'] = str(totalVotesCounted)
        print_data['undecided_constituencies'] = str(C - np.sum(seenWins)) 
        print_data['winner'] = Parties[winner]
        print_data['winner_lcb'] = str(Cl[winner,i])
        print_data['winner_ucb'] = str(Cu[winner,i])
        print_data['second_place'] = Parties[second_place]
        print_data['second_place_lcb'] = str(Cl[second_place,i])
        print_data['second_place_ucb'] = str(Cu[second_place,i])
        f.write(json.dumps(print_data) + '\n')
        print(print_data)

        if term > 0 and not separated:
            separated = True
            separated_i = i
            print_data = {}
            print_data['decided_constituencies'] = str(separated_i)
            print_data['winning_party'] = Parties[winner]
            print_data['seats_won'] = str(seenWins[winner])
            print_data['total_votes'] = str(totalVotesCounted)
            print('Election Complete')
            f.write(json.dumps(print_data) + '\n')
            f.close()
            return separated_i, winner, totalVotesCounted, seenVotes

    return separate_i, winner, totalVotesCounted, seenVotes

if __name__ == "__main__":

    data = "Delhi2015"
    
    alpha = 10**-1

    stoppingC, winner, stoppingVotes, seenVotes = runElection(data, alpha)

    print(stoppingC, winner, stoppingVotes)
