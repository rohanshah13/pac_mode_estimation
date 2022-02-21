import os
import json
import numpy as np
from math import factorial
from scipy.stats import betabinom, beta


def normalize(p):

    return p/np.sum(p)

def sampleBetaBin(n, a, b):
    
    p = np.random.beta(a, b)
    r = np.random.binomial(n, p)

    return r

def scoreCandidate(candidateStats):

    s = 0
    for k in range(len(candidateStats)):
        s += (k+1) * candidateStats[k]

    return s

def scoreCandidateLB(lb, ub, N0):

    x = np.copy(lb)
    
    for k in range(len(lb)):

        N = np.sum(x)
        if N == N0:
            break
        x[k] += min(ub[k]-x[k], N0 - N)

    return scoreCandidate(x)

def scoreCandidateUB(lb, ub, N0):

    x = np.copy(ub)

    for k in range(len(ub)):

        N = np.sum(x)
        if N == N0:
            break
        x[k] -= min(x[k]-lb[k], N-N0)
        
    return scoreCandidate(x)
        

#Calculates the prior posterior ratio of a Beta Binomial Distribution. Beta Binomial distribution - 
#N tosses of a coin whose parameter p is a Random Variable following the Beta distribution
#N0 - total number of votes in the constituency, a = 1, b = 1, t = Total number of votes seen from 
#that constituency, kt = number of succeses(seen votes for a specific party), k = Value of number of party votes at which probability is being evaluated.
def ppr(k, N0, a, b, t, kt):
    assert(kt <= t)
    with np.errstate(divide='ignore', over='ignore'):
            val = betabinom.pmf(k, N0, a, b) / betabinom.pmf(k - kt, N0 - t, a + kt, b + t - kt)
    return val



# binary search
# alpha = error probability
# N0 = total votes in constituency
# a = b = 1 (from prior)
# t = Votes seen in that constituency 
# kb = Votes seen for specific party in the constituency
def binBounds(alpha, N0, a, b, t, kb):

    def f(c):

        return ppr(c, N0, a, b, t, kb) - 1/alpha
    
    def check(c):
        
        if f(c) < 0:
            return c
        else:
            return -1

    def binSearch(l, h, increasing):

        if increasing:
            
            if h - l == 1:
                return l
            
            c = (l+h)//2

            if f(c) < 0:
                l = c
            else:
                h = c

            return binSearch(l, h, True)

        else:
            if h - l == 1:
                return h

            c = (l+h)//2

            if f(c) < 0:
                h = c
            else:
                l = c

            return binSearch(l, h, False)

            

    #find a point inside the confidence interval    
    
    def pointInsideCI(l, h):

        p = 1
        while True:
            #Loop from 1:2^p+1, odd numbers only
            for i in np.arange(1, 2**p + 1, 2):
                # l + ((h-l)*i)//2^p
                #here (N0*i)//2^p
                k = check(l + (h-l) * i // 2**p)

                if k != -1:
                    return k
            p+=1

    k = pointInsideCI(0, N0)

    if f(0) < 0:
        i = 0
    else:
        i = binSearch(0, k, False) 

    if f(N0) < 0:
        j = N0
    else:
        j = binSearch(k, N0, True)
    
    return i, j


def trueWinner(data):

    scores = np.load(data)

    return np.argmax(np.sum(scores, axis = 0))

def getAllPlaces(directory = "data/"):
    
    allPlaces = [x.replace(".csv", "").strip() for x in os.listdir(directory) if x[-4:] == ".csv"]
    allPlaces = [x for x in allPlaces]
    #allPlaces = sorted(allPlaces)
    allPlaces.sort()
    return allPlaces

def getPartyList(file):

    with open(file, 'r') as f:
        return json.load(f)
    

    
