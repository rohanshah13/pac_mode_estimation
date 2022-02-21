import numpy as np

class StakePool(object):

    def __init__(self,num_nodes,frac_byzantine_nodes,num_answers):
        assert frac_byzantine_nodes < 0.5
        self.node_dict = {}
        self.num_nodes = num_nodes

        self.values = np.random.rand(num_answers)
        byzantine_nodes = np.random.choice(num_nodes, int(num_nodes*frac_byzantine_nodes), replace=False)
        for i in range(num_nodes):
            if i in byzantine_nodes:
                self.node_dict[i] = np.random.choice(self.values[1:])
            else:
                self.node_dict[i] = self.values[0]

    def get_execution_set_with_replacement(self,m,first=None):
        es = {}
        if first is not None:
            es[first] = 1
        while True:
            node = np.random.choice(self.num_nodes)
            if node not in es.keys() and len(es.keys()) == m:
                return es, node
            if node not in es.keys():
                es[node] = 1
            else:
                es[node] += 1

    def get_execution_set_without_replacement(self,m):
        nodes = np.random.choice(self.num_nodes, m, replace=False)
        es = {}
        for node in nodes:
            es[node] = 1
        return es   


    def get_execution_set_m_vals(self,m):
        es = {}
        for _ in range(m):
            node = np.random.choice(self.num_nodes)
            if node not in es.keys():
                es[node] = 1
            else:
                es[node] += 1
        return es

    def get_execution_set_m_keys(self,m):
        nodes = np.random.choice(self.num_nodes, m, replace=False)
        es = {}
        for node in nodes:
            es[node] = 1
        return es

    def print_all_node_evalution(self):
        print(self.node_dict)
        return
