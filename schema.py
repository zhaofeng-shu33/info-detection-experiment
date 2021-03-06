'''
This script has two purposes
1. generate or update parameter.json, which can be edited to use the optimal parameter
2. generate latex table, which is used in the paper. The generation can simply read the data
   from parameter.json or run the experiments to get the parameter
'''
import json
import os
import random
import argparse
import pdb

import numpy as np
from tabulate import tabulate

from evaluation import ex

from util import BUILD_DIR, load_parameters, write_parameters


TABLE_NAME = 'id_compare'
DATASET = ['GaussianBlob', 'Moon', 'Lymphography', 'Glass', 'Ionosphere']
METHOD = ['ic', 'lof', 'if', 'ee', 'svm']
METHOD_FULL_NAME = {'ic': 'Info-Detection', 'lof': 'local outlier factor', 
    'if': 'isolation forest', 'ee': 'elliptic envelope', 'svm': 'one class SVM'}
ALG_PARAMS = {'_gamma': 0.1, 'contamination': 0.1, 'n_neighbors': 10, 'affinity': 'rbf'}

def update_json(dic):
    global DATASET, ALG_PARAMS, METHOD
    for dataset in DATASET:
        if not(dic.get(dataset)):
            dic[dataset] = {}
        dic_dataset = dic[dataset]
        for method in METHOD:
            if not(dic_dataset.get(method)):
                dic_dataset[method] = ALG_PARAMS            
             

def run_experiment_matrix(parameter_dic):
    global DATASET
    for dataset, v in parameter_dic.items():
        if DATASET.count(dataset) == 0:
            continue
        for alg, v1 in v.items():
            ex_param_dic = {
                'alg_params': v1,
                'verbose': False,
                'alg': alg,
                'dataset': dataset
            }
            r = ex.run(config_updates= ex_param_dic)
            tpr, tnr = r.result
            if not(v1.get('tpr', False) and v1.get('tnr', False)):
                v1['tpr'] = tpr
                v1['tnr'] = tnr
            elif(abs(tpr - v1['tpr']) > 0.01 or abs(tnr - v1['tnr']) > 0.01):
                v1['tpr'] = tpr
                v1['tnr'] = tnr

def make_table(dic, tb_name, format):
    global METHOD, METHOD_FULL_NAME, DATASET, BUILD_DIR
    table = [[METHOD_FULL_NAME[i]] for i in METHOD]
    for dataset_name, dataset_method in dic.items():
        if DATASET.count(dataset_name) == 0:
            continue
        for index, v in enumerate(dataset_method.values()):
            tpr, tnr = v['tpr'], v['tnr']
            table[index].append('%.1f\\%%/%.1f\\%%'%(100*tpr, 100*tnr))
    _headers = ['TPR/TNR']
    _headers.extend([i for i in DATASET])
    align_list = ['center' for i in range(len(_headers))]
    table_string = tabulate(table, headers = _headers, tablefmt = format, floatfmt='.1f', colalign=align_list)
    if not(os.path.exists(BUILD_DIR)):
        os.mkdir(BUILD_DIR)
        
    if(format == 'latex_raw'):
        table_suffix = '.tex'
        table_string = table_string.replace('100.0','100')
    elif(format == 'html'):
        table_suffix = '.html'
        
    with open(os.path.join(BUILD_DIR, tb_name + table_suffix),'w') as f: 
        f.write(table_string)
        
if __name__ == '__main__':
    dataset_choices = [i for i in DATASET]
    dataset_choices.append('all')
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', default='table', choices=['json', 'table'])
    parser.add_argument('--ignore_computing', help='whether to ignore computing and use ari field in parameter file directly', default=False, type=bool, nargs='?', const=True)
    parser.add_argument('--dataset', help='name of the dataset to fine tuning', default='all', choices=dataset_choices, nargs='+')
    parser.add_argument('--table_format', default='latex_raw', choices=['html', 'latex_raw'])    
    args = parser.parse_args()
    if type(args.dataset) is list:
        DATASET = args.dataset
    if(args.action == 'json'):
        json_obj = load_parameters()
        update_json(json_obj)
        write_parameters(json_obj)
    elif(args.action == 'table'):
        parameter_json = load_parameters()
        if not(args.ignore_computing):
            run_experiment_matrix(parameter_json)
            write_parameters(parameter_json)
        make_table(parameter_json, TABLE_NAME, args.table_format)