import os
import pdb

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sacred import Experiment
from sacred.observers import MongoObserver

from util import TPR_TNR
from util import Lymphography, Glass, Ionosphere
from util import generate_one_blob, generate_two_moon
from util import load_parameters, write_parameters


from info_detection import InfoOutlierDetector

user_name = os.environ.get('SACRED_USER', 'admin')
user_passwd = os.environ.get('SACRED_PASSWD', 'abc')
collection_url = 'mongodb://%s:%s@127.0.0.1/?authSource=user-data'%(user_name, user_passwd)

ex = Experiment('outlier_detection')
if(os.sys.platform != 'win32' and os.environ.get('USE_MONGO')):
    ex.observers.append(MongoObserver.create(
        url=collection_url,
        db_name='sacred'))

 
if(os.path.exists('conf.yaml')):
    ex.add_config('conf.yaml')
else:
    ex_param_dic = {
        'alg_params': {'_gamma': 0.5, 'n_neighbors': 10, 'affinity': 'rbf'},
        'verbose': False,
        'alg': 'ic',
        'dataset': 'GaussianBlob'
    }
    ex.add_config(**ex_param_dic)
@ex.automain
def run(dataset, alg, alg_params, verbose, seed):
    if(dataset == 'Glass'):
        data, labels = Glass()
    elif(dataset == 'Lymphography'):
        data, labels = Lymphography()
    elif(dataset == 'GaussianBlob'):
        data, labels = generate_one_blob()
    elif(dataset == 'Moon'):
        data, labels = generate_two_moon()
    elif dataset == 'Ionosphere':
        data, labels = Ionosphere()
    else:
        raise NameError(dataset + ' dataset name not foud')
    if alg_params.get('seed'):
        inner_seed = alg_params.get('seed')
    else:
        inner_seed = seed
    if(alg == 'ic'):
        alg_instance = InfoOutlierDetector(gamma=alg_params['_gamma'],
            affinity=alg_params['affinity'], n_neighbors=alg_params['n_neighbors'], delta=alg_params['_delta'])
    elif(alg == 'lof'):
        alg_instance = LocalOutlierFactor(n_neighbors=alg_params['n_neighbors'], 
            contamination=alg_params['contamination'])
    elif(alg == 'if'):
        alg_instance = IsolationForest(contamination=alg_params['contamination'], 
            behaviour='new', random_state=inner_seed)
    elif(alg == 'ee'):
        alg_instance = EllipticEnvelope(contamination=alg_params['contamination'], random_state=inner_seed)
    elif(alg == 'svm'):
        alg_instance = OneClassSVM(kernel=alg_params['affinity'], 
            gamma=alg_params['_gamma'], nu=alg_params['contamination'])
    else:
        raise NameError(alg + ' algorithm name not found')
        
    y_predict = alg_instance.fit_predict(data)
    if(alg == 'ic' and verbose):
        print('num of outliers', alg_instance.num_of_outliers)
    tpr, tnr = TPR_TNR(labels, y_predict)
    parameter_json = load_parameters()
    dataset_alg_dic = parameter_json[dataset][alg]
    should_update_parameter_case_1 = (dataset_alg_dic.get('tpr', 0) < 0.9) and (dataset_alg_dic.get('tpr', 0) < tpr)
    should_update_parameter_case_2 = (dataset_alg_dic.get('tpr', 0) > 0.9) and (tpr > 0.9) and (dataset_alg_dic.get('tnr', 0) < tnr)
    if should_update_parameter_case_1 or should_update_parameter_case_2:
        if alg == 'if' or alg == 'ee':
            dataset_alg_dic['seed'] = seed
        dataset_alg_dic['tpr'] = tpr
        dataset_alg_dic['tnr'] = tnr
        dataset_alg_dic.update(alg_params)
        write_parameters(parameter_json)
    ex.log_scalar("tpr", tpr)
    ex.log_scalar("tnr", tnr)
    return (tpr, tnr)    
