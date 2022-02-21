import json
import numpy as np

def write_data(algorithm, num_samples, se, delta, failure_rate, seed, outfile):
    json_data = {}
    json_data['algorithm'] = algorithm
    json_data['num_samples'] = num_samples
    json_data['se'] = se
    json_data['delta'] = delta
    json_data['failure_rate'] = failure_rate
    json_data['seed'] = seed
    with open(outfile, 'a') as f:
        f.write(json.dumps(json_data) + '\n')

def write_data_seed(algorithm, num_samples, delta, num_success, outfile):
    json_data = {}
    json_data['algorithm'] = algorithm
    json_data['samples'] = num_samples
    num_success = [int(x) for x in num_success]
    json_data['success'] = num_success
    # json_data['se'] = se
    json_data['delta'] = delta
    # json_data['failure_rate'] = failure_rate
    with open(outfile, 'a') as f:
        f.write(json.dumps(json_data) + '\n')
