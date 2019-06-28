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
from tabulate import tabulate

from evaluation import ex

BUILD_DIR = 'build'
PARAMETER_FILE = 'parameter.json'
TABLE_NAME = 'id_compare.tex'
DATASET = ['GaussianBlob', 'Moon', 'Lymphography', 'Glass']
METHOD = ['ic', 'lof', 'if']
METHOD_FULL_NAME = {'ic': 'Info-Detection', 'lof': 'local outlier factor', 'if': 'isolation forest'}
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

def run_experiment_matrix(parameter_dic):
    for dataset, v in parameter_dic.items():
        for alg, v1 in v.items():
            ex_param_dic = {
                'alg_params': v1,
                'verbose': False,
                'alg': alg,
                'dataset': dataset
            }
            r = ex.run(config_updates= ex_param_dic)
            tpr, tnr = r.result
            v1['tpr'] = tpr
            v1['tnr'] = tnr

def make_table(dic, tb_name):
    global METHOD, METHOD_FULL_NAME, BUILD_DIR
    table = [[i] for i in dic.keys()]
    for i in table:
        for k, v in dic[i[0]].items():
            tpr, tnr = v['tpr'], v['tnr']
            i.append('%.1f\\%%/%.1f\\%%'%(100*tpr, 100*tnr))
    _headers = ['TPR/FNR']
    _headers.extend([METHOD_FULL_NAME[i] for i in METHOD])
    latex_table_string = tabulate(table, headers = _headers, tablefmt = 'latex_raw', floatfmt='.1f')
    # manually alignment change
    latex_table_string = latex_table_string.replace('llll','lp{2.5cm}p{3cm}p{3cm}')
    with open(os.path.join(BUILD_DIR, tb_name),'w') as f: 
        f.write(latex_table_string)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', default='tex', choices=['json', 'tex'])
    parser.add_argument('--ignore_computing', help='whether to ignore computing and use ari field in parameter file directly', default=False, type=bool, nargs='?', const=True)
    args = parser.parse_args()
    if(args.action == 'json'):
        if (os.path.exists(PARAMETER_FILE)):
            print("parameter file %s exists. Please remove it manually to regenerate."% PARAMETER_FILE)
        else:
            with open(PARAMETER_FILE, 'w') as f:
                json_str = create_json()
                f.write(json_str)
            print('parameter files written to %s' % PARAMETER_FILE)
    elif(args.action == 'tex'):
        if not (os.path.exists(PARAMETER_FILE)):
            print("parameter file %s not exists. Please generate it first."% PARAMETER_FILE)
        else:
            with open(PARAMETER_FILE) as f:
                parameter_json = json.loads(f.read())        
            if not(args.ignore_computing):
                run_experiment_matrix(parameter_json)
                with open(PARAMETER_FILE, 'w') as f:
                    f.write(json.dumps(parameter_json, indent=4))
            make_table(parameter_json, TABLE_NAME)