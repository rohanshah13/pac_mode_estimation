from stakepool import StakePool
from miracle_agent import KL_Agent, Hoeffding_Agent, SPRT_Agent, Lil_UCB_Agent, Beta_CS, PPR_Agent
from scipy.stats import sem
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 11})
import time
import numpy as np
plt.rcParams["figure.autolayout"] = True

fs = []
iterations_sprt = []
nodes_eval_sprt = []
iterations_kl = []
nodes_eval_kl = []
iterations_beta = []
nodes_eval_beta = []
iterations_ppr = []
nodes_eval_ppr = []

accuracy_sprt = []
accuracy_kl = []
accuracy_ppr = []

f = 0.05

beta = 0.01
num_answer = 2 # value of k
m = 20
f_max = 0.1
num_runs = 10000
num_nodes = 1600

def exploration_rate(time,delta):
    return np.log(time*time*np.log2(1/delta)/delta)

for _ in range(8):
    print('=====================================================')
    sprt_agent = SPRT_Agent(f_max)
    beta_cs = Beta_CS(f_max)
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
    accuracy_sprt.append(1 - (correct/num_runs))
    print("SPRT Agent : f = %f done. Accuracy = %f. Average iterations = %f. Average Nodes Evaluated = %f. Time : %f" % (f,correct/num_runs,np.mean(count_itrs),np.mean(count_nodes),time.time()-t1))

    correct = 0.0
    count_itrs = []
    count_nodes = []
    t1 = time.time()
    for i in range(num_runs):
        nodes,val,samples = beta_cs.find_value(sp,m,beta)
        if val == sp.values[0]:
            correct+=1
        count_itrs.append(samples*m)
        count_nodes.append(nodes)
    count_itrs = np.array(count_itrs)
    count_nodes = np.array(count_nodes)
    iterations_beta.append(count_itrs)
    nodes_eval_beta.append(count_nodes)
    accuracy_ppr.append(1 - (correct/num_runs))
    print("PPR-Bernoulli Agent : f = %f done. Accuracy = %f. Average iterations = %f. Average Nodes Evaluated = %f. Time : %f" % (f,correct/num_runs,np.mean(count_itrs),np.mean(count_nodes),time.time()-t1))
        
    fs.append(f)
    f += 0.05

iterations_sprt = np.array(iterations_sprt)
iterations_beta = np.array(iterations_beta)

accuracy_sprt = np.array(accuracy_sprt)
accuracy_ppr = np.array(accuracy_ppr)

plt.errorbar(fs,accuracy_sprt,label='SPRT', marker = 'o')
plt.errorbar(fs,accuracy_ppr,label='PPR-Bernoulli', marker = '*')

plt.xlabel('Fraction of Byzantine Nodes f')
plt.ylabel('Empirical Error')
plt.legend()
plt.savefig('error_plot.png')
