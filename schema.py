'''
This script has two purposes
1. generate parameter.json, which can be edited to use the optimal parameter
2. generate latex table, which is used in the paper. The generation can simply read the data
   from parameter.json or run the experiments to get the parameter
'''
import json
import os
import random
import argparse

import numpy as np

BUILD_DIR = 'build'
PARAMETER_FILE = 'parameter.json'
DATASET = ['GaussianBlob', 'Moon', 'Lymphography', 'Glass']
METHOD = ['ic', 'lof', 'if']
ALG_PARAMS = {'_gamma': 0.1, 'contamination': 0.1, 'n_neighbors': 10, 'affinity': 'rbf'}

def create_json():
    '''update tuning json string
    '''
    global DATASET, DATASET
    dic = {}
    for dataset in DATASET:
        dic[dataset] = {}
        dic_dataset = dic[dataset]
        for method in METHOD:
            dic_dataset[method] = ALG_PARAMS
    return json.dumps(dic, indent=4)                
    
if __name__ == '__main__':

    if not(os.path.exists(BUILD_DIR)):
        os.mkdir(BUILD_DIR)
    parameter_file_path = os.path.join(BUILD_DIR, PARAMETER_FILE)
    if (os.path.exists(parameter_file_path)):
        print("parameter file %s exists. Please remove it manually to regenerate."% parameter_file_path)
    else:
        with open(parameter_file_path, 'w') as f:
            json_str = create_json()
            f.write(json_str)
        print('parameter files written to %s' % parameter_file_path)
            
        