import numpy as np
from util import *
from runElection import *
from runBanditElection import *
from runUniformElection import *
from runDCBElection import *
from runDCBElectionPrior import *
from runDCBElection_PPR1 import *
from runDCBElection_PPR2 import *
from runDCBElection_SE import *
from runDCBElection_GLR import *
from runDCBElection_SEEVE import *
from runUniformElection_SE import *
from runUniformElection_SEEVE import *
from runUniformElection_GLR import *
from runUniformElection1 import *
from runUniformElection_PPR2 import *
from runUniformElection_KLSN import *
from runDCBElection_KLSN import *
from runUniformElection_KLSNEVE import *
from runDCBElection_KLSNEVE import *
import time
import pickle
import sys
import argparse
import os

TRACEFILE = 'results/{}/{}/trace/{}_{}_{}'

def main():

	parser = argparse.ArgumentParser()

	parser.add_argument('--dataset',default='India2014')
	parser.add_argument('--algorithm', default='LUCB')

	args = parser.parse_args()
	## which elections
	data = args.dataset

	## number of runs 
	T = 10

	## mistake probability
	alpha = 10**-2

	## batch size
	batch = 200

	## batch size for initial pulls of each constituency
	init_batch = batch

	algorithm = args.algorithm

	constituenciesDecided = np.zeros(T)
	winners = np.zeros(T)
	totalVotesCounted = np.zeros(T)
	totalLabelledVotesCounted = np.zeros(T)
	time_elapsed = np.zeros(T)
	seenVotes = [None for i in range(T)]
	listVotes = [None for i in range(T)]

	if not os.path.exists('results'):
		os.mkdir('results')
	if not os.path.exists('results/{}'.format(data)):
		os.mkdir('results/{}'.format(data))
	if not os.path.exists('results/{}/{}'.format(data,algorithm)):
		os.mkdir('results/{}/{}'.format(data,algorithm))
	if not os.path.exists('results/{}/{}/trace'.format(data,algorithm)):
		os.mkdir('results/{}/{}/trace'.format(data,algorithm))

	for t in range(0,T):
		tic = time.time()
		np.random.seed(t)
		print("Election  ", t)
		tracefile = TRACEFILE.format(data,algorithm,alpha,batch,t)
		tracefile = 'temp.txt'
		if algorithm == "LUCB":
			constituenciesDecided[t], winners[t], totalVotesCounted[t], seenVotes[t] = runBanditElectionLUCB(data, alpha, tracefile, batch, init_batch)
		elif algorithm == "Uniform":
			constituenciesDecided[t], winners[t], totalVotesCounted[t], seenVotes[t] = runUniformElection(data, alpha, tracefile, batch, init_batch)
		elif algorithm == "RRPPR1VR":
			constituenciesDecided[t], winners[t], totalVotesCounted[t], seenVotes[t], totalLabelledVotesCounted[t] = runUniformElection1(data, alpha, tracefile, batch, init_batch)
		elif algorithm == "RRPPR1V1":
			constituenciesDecided[t], winners[t], totalVotesCounted[t], seenVotes[t], totalLabelledVotesCounted[t] = runUniformElection_PPR2(data, alpha, tracefile, batch, init_batch)
		elif algorithm == "RRA11VR":
			constituenciesDecided[t], winners[t], totalVotesCounted[t], seenVotes[t], totalLabelledVotesCounted[t] = runUniformElection_SE(data, alpha, tracefile, batch, init_batch)
		elif algorithm == "RRA11V1":
			constituenciesDecided[t], winners[t], totalVotesCounted[t], seenVotes[t], totalLabelledVotesCounted[t] = runUniformElection_SEEVE(data, alpha, tracefile, batch, init_batch)
		elif algorithm == "RRKLSN1VR":
			constituenciesDecided[t], winners[t], totalVotesCounted[t], seenVotes[t], totalLabelledVotesCounted[t] = runUniformElection_KLSN(data, alpha, tracefile, batch, init_batch)
		elif algorithm == "RRKLSN1V1":
			constituenciesDecided[t], winners[t], totalVotesCounted[t], seenVotes[t], totalLabelledVotesCounted[t] = runUniformElection_KLSNEVE(data, alpha, tracefile, batch, init_batch)
		elif algorithm == "DCBKLSN1VR":
			constituenciesDecided[t], winners[t], totalVotesCounted[t], seenVotes[t], totalLabelledVotesCounted[t] = runDCBElection_KLSN(data, alpha, tracefile, batch, init_batch)
		elif algorithm == "UGLR":
			constituenciesDecided[t], winners[t], totalVotesCounted[t], seenVotes[t], totalLabelledVotesCounted[t] = runUniformElection_GLR(data, alpha, tracefile, batch, init_batch)
		elif algorithm == "TwoLevelOpinionSurvey":
			constituenciesDecided[t], winners[t], totalVotesCounted[t], seenVotes[t] = runElection(data, alpha, tracefile, batch)
		elif algorithm == "DCB":
			constituenciesDecided[t], winners[t], totalVotesCounted[t], seenVotes[t] = runDCBElection(data, alpha, tracefile, batch, init_batch)
		elif algorithm == "DCB_Prior":
			constituenciesDecided[t], winners[t], totalVotesCounted[t], seenVotes[t] = runDCBElectionPrior(data, alpha, tracefile, batch, init_batch)
		elif algorithm == "DCBPPR1VR":
			constituenciesDecided[t], winners[t], totalVotesCounted[t], seenVotes[t], totalLabelledVotesCounted[t] = runDCBElection_PPR1(data, alpha, tracefile, batch, init_batch)
		elif algorithm == "DCBPPR1V1":
			constituenciesDecided[t], winners[t], totalVotesCounted[t], seenVotes[t], totalLabelledVotesCounted[t] = runDCBElection_PPR2(data, alpha, tracefile, batch, init_batch)
		elif algorithm == "DCBA11VR":
			constituenciesDecided[t], winners[t], totalVotesCounted[t], seenVotes[t], totalLabelledVotesCounted[t] = runDCBElection_SE(data, alpha, tracefile, batch, init_batch)
		elif algorithm == "DCBA11V1":
			constituenciesDecided[t], winners[t], totalVotesCounted[t], seenVotes[t], totalLabelledVotesCounted[t] = runDCBElection_SEEVE(data, alpha, tracefile, batch, init_batch)
		elif algorithm == "DCBGLR":
			constituenciesDecided[t], winners[t], totalVotesCounted[t], seenVotes[t], totalLabelledVotesCounted[t] = runDCBElection_GLR(data, alpha, tracefile, batch, init_batch)
		elif algorithm == "DCBKLSN1V1":
			constituenciesDecided[t], winners[t], totalVotesCounted[t], seenVotes[t], totalLabelledVotesCounted[t] = runDCBElection_KLSNEVE(data, alpha, tracefile, batch, init_batch)	
		# exit()
		toc = time.time()
		time_elapsed[t] = toc - tic

	C = len(seenVotes[0])
	constiVotes = np.zeros((T,C))
	for t in range(T):
		constiVotes[t,:] = np.asarray([sum(x) for x in seenVotes[t]])

	np.save('results/{}/{}/constituenciesPolled_{}_{}_{}'.format(data,algorithm,alpha,batch,T), constituenciesDecided)

	np.save('results/{}/{}/winnerIDs_{}_{}_{}'.format(data,algorithm,alpha,batch,T), winners)

	np.save('results/{}/{}/totalVotesCounted_{}_{}_{}'.format(data,algorithm,alpha,batch,T), totalVotesCounted)

	np.save('results/{}/{}/votesSeen_{}_{}_{}.npy'.format(data,algorithm,alpha,batch,T), constiVotes)

	if algorithm in ['RRPPR1VR', 'RRPPR1V1', 'RRA11VR', 'UGLR', 'DCBPPR1VR', 'DCBPPR1V1', 'DCBA11VR', 'DCBGLR', 'DCBA11V1', 'RRA11V1', 'RRKLSN1VR', 'DCBKLSN1VR', 'RRKLSN1V1', 'DCBKLSN1V1']:
		np.save('results/{}/{}/totalLabelledVotesCounted_{}_{}_{}'.format(data,algorithm,alpha,batch,T), totalLabelledVotesCounted)		

	print(f"Algorithm = {algorithm}, alpha = {alpha}, batch = {batch}, T = {T}")
	print("Constituencies decided = ", np.mean(constituenciesDecided), " +- ", np.std(constituenciesDecided)/np.sqrt(T))
	print("Votes counted (unlabelled)= ", np.mean(totalVotesCounted), " +- ", np.std(totalVotesCounted)/np.sqrt(T))
	if algorithm in ['RRPPR1VR', 'RRPPR1V1', 'RRA11VR', 'UGLR', 'DCBPPR1VR', 'DCBPPR1V1', 'DCBA11VR', 'DCBGLR', 'DCBA11V1', 'RRA11V1', 'RRKLSN1VR', 'DCBKLSN1VR', 'RRKLSN1V1', 'DCBKLSN1V1']:
		print("Votes counted (labelled)= ", np.mean(totalLabelledVotesCounted), " +- ", np.std(totalLabelledVotesCounted)/np.sqrt(T))
	
	print("Time elapsed: {} +- {}".format(np.mean(time_elapsed), np.std(time_elapsed) / np.sqrt(T)))
if __name__ == "__main__":

	main()
