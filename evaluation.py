import os
import pdb

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sacred import Experiment
from sacred.observers import MongoObserver

from util import TPR_TNR
from util import Lymphography, Glass

from info_detection import InfoOutlierDetector

user_name = os.environ.get('SACRED_USER', 'admin')
user_passwd = os.environ.get('SACRED_PASSWD', 'abc')
collection_url = 'mongodb://%s:%s@127.0.0.1/?authSource=user-data'%(user_name, user_passwd)

ex = Experiment('outlier_detection')
if(os.sys.platform != 'win32'):
    ex.observers.append(MongoObserver.create(
        url=collection_url,
        db_name='sacred'))

 

@ex.config
def cfg():
    alg_params = {'_gamma' : 0.03, 'contamination' : 0.041, 'n_neighbors' : 62, 'affinity' : ['rbf', 'nearest_neighbors']}
    alg = 'ic' # choices from ['ic', 'lof']
    verbose = True   
    dataset = 'Lymphography'
 
@ex.automain
def run(dataset, alg, alg_params, verbose):
    if(dataset == 'Glass'):
        data, labels = Glass()
    elif(dataset == 'Lymphography'):
        data, labels = Lymphography()
    else:
        raise NameError(dataset + ' dataset name not foud')

    if(alg == 'ic'):
        alg_instance = InfoOutlierDetector(gamma=alg_params['_gamma'],
            affinity=alg_params['affinity'])
    elif(alg == 'lof'):
        alg_instance = LocalOutlierFactor(n_neighbors=alg_params['n_neighbors'], 
            contamination=alg_params['contamination'])
    else:
        raise NameError(alg + ' algorithm name not found')
    y_predict = alg_instance.fit_predict(data)
    if(alg == 'ic' and verbose):
        print(alg_instance.partition_num_list)
    tpr, tnr = TPR_TNR(labels, y_predict)
    ex.log_scalar("tpr", tpr)        
    ex.log_scalar("tnr", tnr)    
    return (tpr, tnr)    
