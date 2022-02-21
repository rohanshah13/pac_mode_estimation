from stakepool import StakePool
from miracle_agent import KL_Agent, Hoeffding_Agent, SPRT_Agent, Lil_UCB_Agent, Beta_CS, PPR_Agent, PPR_1v1, PPR_1vr, PPR_Ada

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 11})
import time
import numpy as np
plt.rcParams["figure.autolayout"] = True

np.random.seed(1)

fs = []
iterations_sprt = []
nodes_eval_sprt = []
iterations_ppr1v1 = []
nodes_eval_ppr1v1 = []
iterations_ppr1vr = []
nodes_eval_ppr1vr = []
iterations_pprada = []
nodes_eval_pprada = []

accuracy_sprt = []
accuracy_kl = []
accuracy_ppr = []

f = 0.05
beta = 0.01
num_answer = 4
m = 20
f_max = 0.4
num_runs = 100
num_nodes = 1600

def exploration_rate(time,delta):
    return np.log(time*time*np.log2(1/delta)/delta)

for _ in range(7):
    print('=====================================================')
    sprt_agent = SPRT_Agent(f_max)
    ppr1v1_cs = PPR_1v1(f_max, num_answer)
    ppr1vr_cs = PPR_1vr(f_max, num_answer)
    pprada_cs = PPR_Ada(f_max)
    sp = StakePool(num_nodes,f,num_answer)

    correct = 0.0
    count_itrs = []
    count_nodes = []
    t1 = time.time()
    for i in range(num_runs):
        nodes,val,samples = sprt_agent.find_value(sp,m,beta)
        if val == sp.values[0]:
            correct+=1
        count_itrs.append(samples * m)
        count_nodes.append(nodes)
    count_itrs = np.array(count_itrs)
    count_nodes = np.array(count_nodes)
    iterations_sprt.append(count_itrs)
    nodes_eval_sprt.append(count_nodes)
    print("SPRT Agent : f = %f done. Accuracy = %f. Average iterations = %f. Average Nodes Evaluated = %f. Time : %f" % (f,correct/num_runs,np.mean(count_itrs),np.mean(count_nodes),time.time()-t1))

    correct = 0.0
    count_itrs = []
    count_nodes = []
    t1 = time.time()
    for i in range(num_runs):
        nodes,val,samples = ppr1v1_cs.find_value(sp,m,beta)
        if val == sp.values[0]:
            correct+=1
        count_itrs.append(samples*m)
        count_nodes.append(nodes)
    count_itrs = np.array(count_itrs)
    count_nodes = np.array(count_nodes)
    iterations_ppr1v1.append(count_itrs)
    nodes_eval_ppr1v1.append(count_nodes)
    print("PPR-1v1 Agent : f = %f done. Accuracy = %f. Average iterations = %f. Average Nodes Evaluated = %f. Time : %f" % (f,correct/num_runs,np.mean(count_itrs),np.mean(count_nodes),time.time()-t1))

    correct = 0.0
    count_itrs = []
    count_nodes = []
    t1 = time.time()
    for i in range(num_runs):
        nodes,val,samples = ppr1vr_cs.find_value(sp,m,beta)
        if val == sp.values[0]:
            correct+=1
        count_itrs.append(samples*m)
        count_nodes.append(nodes)
    count_itrs = np.array(count_itrs)
    count_nodes = np.array(count_nodes)
    iterations_ppr1vr.append(count_itrs)
    nodes_eval_ppr1vr.append(count_nodes)
    print("PPR-1vr Agent : f = %f done. Accuracy = %f. Average iterations = %f. Average Nodes Evaluated = %f. Time : %f" % (f,correct/num_runs,np.mean(count_itrs),np.mean(count_nodes),time.time()-t1))

    correct = 0.0
    count_itrs = []
    count_nodes = []
    t1 = time.time()
    for i in range(num_runs):
        nodes,val,samples = pprada_cs.find_value(sp,m,beta)
        if val == sp.values[0]:
            correct+=1
        count_itrs.append(samples*m)
        count_nodes.append(nodes)
    count_itrs = np.array(count_itrs)
    count_nodes = np.array(count_nodes)
    iterations_pprada.append(count_itrs)
    nodes_eval_pprada.append(count_nodes)
    print("PPR-Adaptive Agent : f = %f done. Accuracy = %f. Average iterations = %f. Average Nodes Evaluated = %f. Time : %f" % (f,correct/num_runs,np.mean(count_itrs),np.mean(count_nodes),time.time()-t1))

    fs.append(f)
    f += 0.05

iterations_sprt = np.array(iterations_sprt)
iterations_ppr1v1 = np.array(iterations_ppr1v1)
iterations_ppr1vr = np.array(iterations_ppr1vr)
iterations_pprada = np.array(iterations_pprada)

accuracy_sprt = np.array(accuracy_sprt)
accuracy_ppr = np.array(accuracy_ppr)

from scipy.stats import sem

mean_sprt = np.mean(iterations_sprt,axis=1)
std_sprt = sem(iterations_sprt,axis=1)
mean_1v1 = np.mean(iterations_ppr1v1,axis=1)
std_1v1 = sem(iterations_ppr1v1,axis=1)
mean_1vr = np.mean(iterations_ppr1vr,axis=1)
std_1vr = sem(iterations_ppr1vr,axis=1)
mean_ada = np.mean(iterations_pprada,axis=1)
std_ada = sem(iterations_pprada,axis=1)

plt.errorbar(fs,mean_ada,yerr = std_ada,label='PPR-Adaptive', marker = '^')
plt.errorbar(fs,mean_1vr,yerr = std_1vr,label='PPR-1vr', marker = 's')
plt.errorbar(fs,mean_1v1,yerr = std_1v1,label='PPR-1v1', marker = '*')
plt.errorbar(fs,mean_sprt,yerr = std_sprt,label='SPRT', marker = 'o')
plt.xlabel('Fraction of Byzantine Nodes f')
plt.ylabel('Sample Complexity')
plt.legend()
plt.savefig('sample_complexity.png')

