import numpy as np
from scipy import stats
from scipy.special import comb
from scipy.stats import beta
from collections import defaultdict


class PPR_Ada(object):
    def __init__(self,f_max):
        self.time = 0.0
        self.nodes_evaluated = []
        self.f_max = f_max
        self.value_counts = {}
        self.node_dict = {}
        self.ci_list = defaultdict()
        self.mapping = defaultdict()
        self.num_answer = 1

        assert f_max < 0.5


    def reset(self):
        self.time = 0.0
        self.nodes_evaluated = []
        self.value_counts = {}
        self.ci_list = defaultdict()
        self.num_answer = 1

    def binary_confidence_sequence(self,params_final, step_size=0.01, alpha=0.05, e=1e-12):
        '''
            params_init: parameters of the prior beta distribution (list)
            params_final: parameters of the posterior beta distribution (list)
            step_size: For searching the parameter space
            alpha: error probability
        '''

        # possible p values
        p_vals = np.linspace(0, 1, num=int(1 / step_size) + 1)
        indices = np.arange(len(p_vals))

        # computation of prior
        # log_prior_0 = beta.logpdf(p_vals, params_init[0], params_init[1])
        # log_prior_1 = beta.logpdf(p_vals, params_init[1], params_init[0])
        log_prior_0 = 0
        log_prior_1 = 0

        # computation of posterior
        log_posterior_0 = beta.logpdf(p_vals, params_final[0], params_final[1])
        log_posterior_1 = beta.logpdf(p_vals, params_final[1], params_final[0])

        # martingale computation
        log_martingale_0 = log_prior_0 - log_posterior_0
        log_martingale_1 = log_prior_1 - log_posterior_1

        # Confidence intervals
        ci_condition_0 = log_martingale_0 < np.log(1 / alpha)
        ci_condition_1 = log_martingale_1 < np.log(1 / alpha)
        
        ci_indices_0 = np.copy(indices[ci_condition_0])
        ci_indices_1 = np.copy(indices[ci_condition_1])
        return [p_vals[np.min(ci_indices_0)], p_vals[np.max(ci_indices_0)]], [p_vals[np.min(ci_indices_1)], p_vals[np.max(ci_indices_1)]]

        
    def checkConverged(self,beta,m):

        for key in self.value_counts.keys():
            
            self.ci_list[key], _ = self.binary_confidence_sequence([self.value_counts[key]+1, self.time - self.value_counts[key]+1 ], alpha = beta/self.mapping[key]**2)


        # finding the maximum lower bound for stopping condition
        lower_bounds = [ci[0] for ci in self.ci_list.values()]
        assumed_mode_index = np.argmax(lower_bounds)
        assumed_mode = list(self.value_counts.keys())[assumed_mode_index]
        terminate = True

        for key in self.value_counts.keys():
            if key != assumed_mode and self.ci_list[key][1] > self.ci_list[assumed_mode][0]:
                terminate = False
                break

        if terminate:
            self.output = assumed_mode
            return True
        return False

    def find_value(self,sp,m,beta):
        self.reset()
        self.iterations = 0
        ci_table = np.zeros((sp.num_nodes,sp.num_nodes,2))
        ci_table[:,:,1] =1
        first = None
        
        while self.iterations==0 or not self.checkConverged(beta,m):

            es, first = sp.get_execution_set_with_replacement(m, first)
            self.time = self.time + sum(es.values())
            for key in es.keys():
                if key not in self.nodes_evaluated:
                    self.nodes_evaluated.append(key)
                node_val = sp.node_dict[key]
                node_count = es[key]
                if node_val not in self.value_counts.keys():
                    self.value_counts[node_val] = node_count
                    self.mapping[node_val] = (self.num_answer+1)*3.14/np.sqrt(6)
                    self.num_answer += 1
                else:
                    self.value_counts[node_val] += node_count
            self.iterations += 1

        return len(self.nodes_evaluated),self.output,self.iterations

class PPR_1vr(object):
    def __init__(self,f_max, num_ans):
        self.time = 0.0
        self.nodes_evaluated = []
        self.f_max = f_max
        self.value_counts = {}
        self.node_dict = {}
        self.num_ans = num_ans
        self.params_init = [[1, 1] for index in range(num_ans)]
        self.ci_list = defaultdict()

        assert f_max < 0.5


    def reset(self):
        self.time = 0.0
        self.nodes_evaluated = []
        self.value_counts = {}
        self.ci_list = defaultdict()

    def binary_confidence_sequence(self,params_final, step_size=0.01, alpha=0.05, e=1e-12):
        '''
            params_init: parameters of the prior beta distribution (list)
            params_final: parameters of the posterior beta distribution (list)
            step_size: For searching the parameter space. Default is 0.01, needs to be smaller if probability values are not well separated.
            alpha: error probability
        '''

        # possible p values
        params_init = self.params_init
        p_vals = np.linspace(0, 1, num=int(1 / step_size) + 1)
        indices = np.arange(len(p_vals))

        # computation of prior
        # log_prior_0 = beta.logpdf(p_vals, params_init[0], params_init[1])
        # log_prior_1 = beta.logpdf(p_vals, params_init[1], params_init[0])
        log_prior_0 = 0
        log_prior_1 = 0

        # computation of posterior
        log_posterior_0 = beta.logpdf(p_vals, params_final[0], params_final[1])
        log_posterior_1 = beta.logpdf(p_vals, params_final[1], params_final[0])

        # martingale computation
        log_martingale_0 = log_prior_0 - log_posterior_0
        log_martingale_1 = log_prior_1 - log_posterior_1

        # Confidence intervals
        ci_condition_0 = log_martingale_0 < np.log(1 / alpha)
        ci_condition_1 = log_martingale_1 < np.log(1 / alpha)
        
        ci_indices_0 = np.copy(indices[ci_condition_0])
        ci_indices_1 = np.copy(indices[ci_condition_1])
        return [p_vals[np.min(ci_indices_0)], p_vals[np.max(ci_indices_0)]], [p_vals[np.min(ci_indices_1)], p_vals[np.max(ci_indices_1)]]

        
    def checkConverged(self,beta,m):

        for key in self.value_counts.keys():
            
            self.ci_list[key], _ = self.binary_confidence_sequence([self.value_counts[key]+1, self.time - self.value_counts[key]+1 ], alpha = beta/self.num_ans)


        # finding the maximum lower bound for stopping condition
        lower_bounds = [ci[0] for ci in self.ci_list.values()]
        assumed_mode_index = np.argmax(lower_bounds)
        assumed_mode = list(self.value_counts.keys())[assumed_mode_index]
        terminate = True

        for key in self.value_counts.keys():
            if key != assumed_mode and self.ci_list[key][1] > self.ci_list[assumed_mode][0]:
                terminate = False
                break

        if terminate:
            self.output = assumed_mode
            return True
        return False

    def find_value(self,sp,m,beta):
        self.reset()
        self.iterations = 0
        ci_table = np.zeros((sp.num_nodes,sp.num_nodes,2))
        ci_table[:,:,1] =1
        first = None
        while self.iterations==0 or not self.checkConverged(beta,m):

            es, first = sp.get_execution_set_with_replacement(m, first)
            self.time = self.time + sum(es.values())
            for key in es.keys():
                if key not in self.nodes_evaluated:
                    self.nodes_evaluated.append(key)
                node_val = sp.node_dict[key]
                node_count = es[key]
                if node_val not in self.value_counts.keys():
                    self.value_counts[node_val] = node_count
                else:
                    self.value_counts[node_val] += node_count
            self.iterations += 1

        return len(self.nodes_evaluated),self.output,self.iterations

class PPR_1v1(object):
    def __init__(self,f_max, num_ans):
        self.time = 0.0
        self.nodes_evaluated = []
        self.f_max = f_max
        self.value_counts = {}
        self.node_dict = {}
        self.num_ans = num_ans
        assert f_max < 0.5


    def reset(self):
        self.time = 0.0
        self.nodes_evaluated = []
        self.value_counts = {}

    def checkConverged(self,beta,m):
        self.value_counts =  dict(sorted(self.value_counts.items(), key=lambda x:x[1], reverse=True))
        if(len(self.value_counts)==1): 
            self.output = list(self.value_counts.keys())[0]
            return True

        top1, top2 = list(self.value_counts.keys())[0], list(self.value_counts.keys())[1]
        if(stats.beta.pdf(0.5, self.value_counts[top1]+1, self.value_counts[top2] + 1) < beta/(self.num_ans - 1)):
            self.output = top1
            return True
        return False


    def find_value(self,sp,m,beta):
        self.reset()
        self.iterations = 0
        ci_table = np.zeros((sp.num_nodes,sp.num_nodes,2))
        ci_table[:,:,1] =1
        first = None
        while self.iterations==0 or not self.checkConverged(beta,m):

            es, first = sp.get_execution_set_with_replacement(m, first)
            self.time = self.time + sum(es.values())
            for key in es.keys():
                if key not in self.nodes_evaluated:
                    self.nodes_evaluated.append(key)
                node_val = sp.node_dict[key]
                node_count = es[key]
                if node_val not in self.value_counts.keys():
                    self.value_counts[node_val] = node_count
                else:
                    self.value_counts[node_val] += node_count
            self.iterations += 1

        return len(self.nodes_evaluated),self.output,self.iterations


class Beta_CS(object):
    def __init__(self,f_max):
        self.time = 0.0
        self.nodes_evaluated = []
        self.f_max = f_max
        self.value_counts = {}
        self.node_dict = {}
        assert f_max < 0.5

    def binary_confidence_sequence(self, step_size=0.01, alpha=0.05, e=1e-12):
        # possible p values
        if(self.iterations==0): return False
        if(len(list(self.value_counts.values()))<2): return False
        p_vals = np.linspace(0.0, 1.0, num=int(100) + 1)
        indices = np.arange(len(p_vals))

        log_prior_0 = beta.logpdf(p_vals, 1, 1)
        log_prior_1 = beta.logpdf(p_vals, 1, 1)

        # computation of posterior
        log_posterior_0 = beta.logpdf(p_vals, list(self.value_counts.values())[0], list(self.value_counts.values())[1])
        log_posterior_1 = beta.logpdf(p_vals, list(self.value_counts.values())[1], list(self.value_counts.values())[0])

        # martingale computation
        log_martingale_0 = log_prior_0 - log_posterior_0
        log_martingale_1 = log_prior_1 - log_posterior_1

        # Confidence intervals
        ci_condition_0 = log_martingale_0 < np.log(1 / alpha)
        ci_condition_1 = log_martingale_1 < np.log(1 / alpha)
        
        ci_indices_0 = np.copy(indices[ci_condition_0])
        ci_indices_1 = np.copy(indices[ci_condition_1])
        return [p_vals[np.min(ci_indices_0)], p_vals[np.max(ci_indices_0)]], [p_vals[np.min(ci_indices_1)], p_vals[np.max(ci_indices_1)]]


    def checkConverged(self,beta,m):
        flag = 0
        for key in self.value_counts.keys():


            log_posterior = stats.beta.logpdf(0.5, self.value_counts[key]+1, self.time - self.value_counts[key] + 1)
            if(-1*log_posterior > np.log(1/beta) ):
                flag = 1
                break

        if(flag==1):
            if(len(self.value_counts.keys())==1): 
                key1 = list(self.value_counts.keys())[0]
                self.output = key1
                return True
            key1, key2 = self.value_counts.keys()
            if(self.value_counts[key1]>self.value_counts[key2]): self.output = key1
            else: self.output = key2
            return True

        return False

    def reset(self):
        self.time = 0.0
        self.nodes_evaluated = []
        self.value_counts = {}

    def find_value(self,sp,m,beta):
        self.reset()
        self.iterations = 0
        first = None
        while not self.checkConverged(beta,m) or self.iterations==0:
            es, first = sp.get_execution_set_with_replacement(m, first)
            self.time = self.time + sum(es.values())
            for key in es.keys():
                if key not in self.nodes_evaluated:
                    self.nodes_evaluated.append(key)
                node_val = sp.node_dict[key]
                node_count = es[key]
                if node_val not in self.value_counts.keys():
                    self.value_counts[node_val] = node_count
                else:
                    self.value_counts[node_val] += node_count
            self.iterations += 1
        return len(self.nodes_evaluated),self.output,self.iterations

class PPR_Agent(object):

    def __init__(self,f_max):
        self.time = 0.0
        self.value_counts_list = {}
        self.nodes_evaluated = []
        self.f_max = f_max
        
        assert f_max < 0.5

    def checkConverged(self,beta,es_size,N):
        a = comb(N,es_size)

        for key in self.value_counts.keys():

            if key not in self.posteriors.keys():
              posterior = np.ones(N+1)
              for i in range(len(self.value_counts[key])):
                  for j in range(N+1):
                    posterior[j] *= comb(j,self.value_counts[key][i])*comb(N-j,es_size - self.value_counts[key][i])/a
              posterior = posterior/posterior.sum()
              self.posteriors[key] = posterior
            else:
              posterior = self.posteriors[key]
              for j in range(N+1):
                  posterior[j] *= comb(j,self.value_counts[key][-1])*comb(N-j,es_size - self.value_counts[key][-1])/a
              posterior = posterior/posterior.sum()
              self.posteriors[key] = posterior

            if (np.argwhere((self.posteriors[key])*(N+1) > beta).squeeze(1) > self.f_max*N).all():
                self.output = key
                return True
        return False

    def reset(self):
        self.time = 0.0
        self.nodes_evaluated = []
        self.value_counts = {}

    def find_value(self,sp,m,beta):
        self.reset()
        iterations = 0
        self.posteriors = {}
        while not self.checkConverged(beta,m,sp.num_nodes):
            es = sp.get_execution_set_m_keys(m)

            self.time = self.time + m
            iteration_val_count = {}

            for key in es.keys():
                if key not in self.nodes_evaluated:
                    self.nodes_evaluated.append(key)
                node_val = sp.node_dict[key]
                if node_val not in iteration_val_count.keys():
                    iteration_val_count[node_val] = 1
                else:
                    iteration_val_count[node_val] += 1

            for key in iteration_val_count.keys():
                if key not in self.value_counts.keys():
                    self.value_counts[key] = [0]*iterations + [iteration_val_count[key]]
                else:
                    self.value_counts[key].append(iteration_val_count[key])

            for key in self.value_counts.keys():
                if key not in iteration_val_count.keys():
                    self.value_counts[key].append(0)

            iterations += 1
            # print(iterations)
        return len(self.nodes_evaluated),self.output,iterations


class KL_Agent(object):

    def __init__(self,f_max,exploration_rate):
        self.time = 0.0
        self.f_max = f_max
        self.value_counts = {}
        self.exploration_rate = exploration_rate
        self.node_dict = {}
        self.nodes_evaluated = []
        assert f_max < 0.5

    def checkConverged(self,beta,m):
        for key in self.value_counts.keys():
            if self.get_kl_lower_bound(self.value_counts[key]/self.time,self.exploration_rate(self.iterations,beta)) > self.f_max:
                self.output = key
                return True
        return False

    def KL_div(self,a, b):
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

    def get_kl_lower_bound(self,p,beta):
        lo = 0
        hi = p
        q = (lo+hi)/2
        lhs = self.KL_div(p,q)*self.time

        while abs(beta-lhs) > 1e-6:
            if abs(hi-lo) < 1e-10:
                break
            if lhs > beta:
                lo = q
            else:
                hi = q
            q = (lo+hi)/2
            lhs = self.KL_div(p,q)*self.time
        return q

    def reset(self):
        self.time = 0.0
        self.nodes_evaluated = []
        self.value_counts = {}

    def find_value(self,sp,m,beta):
        self.reset()
        self.iterations = 0
        first = None
        while not self.checkConverged(beta,m):
            es, first = sp.get_execution_set_with_replacement(m, first)
            self.time = self.time + sum(es.values())
            for key in es.keys():
                if key not in self.nodes_evaluated:
                    self.nodes_evaluated.append(key)
                node_val = sp.node_dict[key]
                node_count = es[key]
                if node_val not in self.value_counts.keys():
                    self.value_counts[node_val] = node_count
                else:
                    self.value_counts[node_val] += node_count
            self.iterations += 1
        return len(self.nodes_evaluated),self.output,self.iterations

class Hoeffding_Agent(object):

    def __init__(self,f_max,exploration_rate):
        self.time = 0.0
        self.f_max = f_max
        self.value_counts = {}
        self.exploration_rate = exploration_rate
        self.node_dict = {}
        self.nodes_evaluated = []
        assert f_max < 0.5

    def checkConverged(self,beta,m):
        for key in self.value_counts.keys():
            if self.get_hoeffding_lower_bound(self.value_counts[key]/self.time,self.exploration_rate(self.iterations,beta)) > self.f_max:
                self.output = key
                return True
        return False

    def get_hoeffding_lower_bound(self,p,beta):
        return p - np.sqrt(beta/(2*self.time))


    def reset(self):
        self.time = 0.0
        self.nodes_evaluated = []
        self.value_counts = {}

    def find_value(self,sp,m,beta):
        self.reset()
        self.iterations = 0
        first = None
        while not self.checkConverged(beta,m):
            es, first = sp.get_execution_set_with_replacement(m, first)
            self.time = self.time + sum(es.values())
            for key in es.keys():
                if key not in self.nodes_evaluated:
                    self.nodes_evaluated.append(key)
                node_val = sp.node_dict[key]
                node_count = es[key]
                if node_val not in self.value_counts.keys():
                    self.value_counts[node_val] = node_count
                else:
                    self.value_counts[node_val] += node_count
            self.iterations += 1
        return len(self.nodes_evaluated),self.output,self.iterations

class Lil_UCB_Agent(object):

    def __init__(self,f_max,epsilon,beta):
        self.time = 0.0
        self.f_max = f_max
        self.value_counts = {}
        self.node_dict = {}
        self.epsilon = epsilon
        self.nodes_evaluated = []
        self.beta = beta
        assert f_max < 0.5

    def checkConverged(self,delta,m):
        for key in self.value_counts.keys():
            if self.get_lil_ucb_lower_bound(self.value_counts[key]/self.time,self.time,delta,m) > self.f_max:
                self.output = key
                return True
        return False

    def get_lil_ucb_lower_bound(self,p,time,delta,m):
        diff = (1+self.epsilon)*time/m + 2
        diff = np.log(diff)
        diff = diff/delta
        diff = np.log(diff)
        diff = 2*(1+self.epsilon)*diff
        diff = diff/time
        diff = np.sqrt(diff)
        diff = (1+self.beta)*(1+np.sqrt(self.epsilon))*diff
        return p - diff


    def reset(self):
        self.time = 0.0
        self.nodes_evaluated = []
        self.value_counts = {}

    def find_value(self,sp,m,beta):
        self.reset()
        iterations = 0
        while not self.checkConverged(beta,m):
            es = sp.get_execution_set_m_vals(m)
            self.time = self.time + sum(es.values())
            for key in es.keys():
                if key not in self.nodes_evaluated:
                    self.nodes_evaluated.append(key)
                node_val = sp.node_dict[key]
                node_count = es[key]

                if node_val not in self.value_counts.keys():
                    self.value_counts[node_val] = node_count
                else:
                    self.value_counts[node_val] += node_count
            iterations += 1
        return len(self.nodes_evaluated),self.output,iterations


class SPRT_Agent(object):

    def __init__(self,f_max):
        self.time = 0.0
        self.value_counts_list = {}
        self.nodes_evaluated = []
        self.f_max = f_max
        self.log_likelihood = {}
        assert f_max < 0.5

    def checkConverged(self,threshold,es_size):

        for key in self.value_counts.keys():
            if key not in self.log_likelihood.keys():
              log_likelihood = 0.0
              for i in range(len(self.value_counts[key])):
                  log_likelihood = log_likelihood + (2*self.value_counts[key][i] - es_size)*es_size
              self.log_likelihood[key] = log_likelihood

            else:
              log_likelihood = self.log_likelihood[key] + (2*self.value_counts[key][-1] - es_size)*es_size
              self.log_likelihood[key] = log_likelihood
            if log_likelihood > threshold :
                # print(log_likelihood, threshold)
                self.output = key
                return True
        return False

    def reset(self):
        self.log_likelihood = {}
        self.time = 0.0
        self.nodes_evaluated = []
        self.value_counts = {}

    def find_value(self,sp,m,beta):
        self.reset()
        iterations = 0
        q = m*1.0/sp.num_nodes
        threshold = np.log((1-beta)/beta)*(2*q*(1-q)*sp.num_nodes*(1-self.f_max)*self.f_max)/(1-2*self.f_max)
        first  = None

        while not self.checkConverged(threshold,m):
            es, first = sp.get_execution_set_with_replacement(m, first)
            # es = sp.get_execution_set_m_keys(m)
            self.time = self.time + m
            iteration_val_count = {}

            for key in es.keys():
                if key not in self.nodes_evaluated:
                    self.nodes_evaluated.append(key)
                node_val = sp.node_dict[key]
                if node_val not in iteration_val_count.keys():
                    iteration_val_count[node_val] = 1
                else:
                    iteration_val_count[node_val] += 1

            for key in iteration_val_count.keys():
                if key not in self.value_counts.keys():
                    self.value_counts[key] = [0]*iterations + [iteration_val_count[key]]
                else:
                    self.value_counts[key].append(iteration_val_count[key])

            for key in self.value_counts.keys():
                if key not in iteration_val_count.keys():
                    self.value_counts[key].append(0)

            iterations += 1
        return len(self.nodes_evaluated),self.output,iterations
