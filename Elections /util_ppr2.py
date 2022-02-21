import os
import json
import numpy as np
from math import factorial
from scipy.stats import betabinom, beta

def ppr2(x, a, b, t, kt):
    val = beta.logpdf(x, a, b) - beta.logpdf(x, a + kt, b + t - kt)
    return val


# def binBounds2(alpha, a, b, t, kb):
#     def f(c):
#         return ppr2(c, a, b, t, kb) - np.log(1/alpha)

#     def check(c):
        
#         if f(c) < 0:
#             return c
#         else:
#             return -1

#     def binSearch(l, h, increasing):

#         if increasing:
            
#             if h - l < 1e-6:
#                 return l
            
#             c = (l+h)/2

#             if f(c) < 0:
#                 l = c
#             else:
#                 h = c

#             return binSearch(l, h, True)

#         else:
#             if h - l < 1e-6:
#                 return h

#             c = (l+h)/2

#             if f(c) < 0:
#                 h = c
#             else:
#                 l = c

#             return binSearch(l, h, False)

            

#     #find a point inside the confidence interval    
    
#     def pointInsideCI(l, h):

#         p = 1
#         p_val = np.
#         while True:
#             #Loop from 1:2^p+1, odd numbers only
#             for i in np.arange(1, 2**p + 1, 2):
#                 # l + ((h-l)*i)//2^p
#                 #here (N0*i)//2^p
#                 k = check( i / 2**p)

#                 if k != -1:
#                     return k
#             p+=1

#     x = pointInsideCI(0, 1)

#     if f(0) < 0:
#         i = 0
#     else:
#         i = binSearch(0, x, False) 

#     if f(1) < 0:
#         j = 1
#     else:
#         j = binSearch(x, 1, True)
    
#     return i, j

def binBounds2(alpha, a, b, t, kb):

    # possible p values
    p_vals = np.linspace(0, 1, num=int(1 / 0.001) + 1)
    indices = np.arange(len(p_vals))

    # computation of prior
    log_prior_0 = beta.logpdf(p_vals, a, b)

    # computation of posterior
    log_posterior_0 = beta.logpdf(p_vals, a + kb, b + t - kb)

    # martingale computation
    log_martingale_0 = log_prior_0 - log_posterior_0

    # Confidence intervals
    ci_condition_0 = log_martingale_0 < np.log(1 / alpha)
    
    ci_indices_0 = np.copy(indices[ci_condition_0])
    return [p_vals[np.min(ci_indices_0)], p_vals[np.max(ci_indices_0)]]

def getAllPlaces(directory = "data/"):
    
    allPlaces = [x.replace(".csv", "").strip() for x in os.listdir(directory) if x[-4:] == ".csv"]
    allPlaces = [x for x in allPlaces]
    #allPlaces = sorted(allPlaces)
    allPlaces.sort()
    return allPlaces

def getPartyList(file):
    with open(file, 'r') as f:
        return json.load(f)

def checkTerm(alpha, a, b, t, kb):
    ppr_val = ppr2(0.5, a, b, t, kb)
    if ppr_val - np.log(1/alpha) < 0:
        return False
    return True