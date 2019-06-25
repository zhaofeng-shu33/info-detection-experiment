import os
import pdb

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import scale
from sklearn.neighbors import LocalOutlierFactor
from sacred import Experiment
from sacred.observers import MongoObserver

from info_detection import InfoOutlierDetector
from uci_glass_outlier import fetch_uci_glass_outlier

user_name = os.environ.get('SACRED_USER', 'admin')
user_passwd = os.environ.get('SACRED_PASSWD', 'abc')
collection_url = 'mongodb://%s:%s@127.0.0.1/?authSource=user-data'%(user_name, user_passwd)

ex = Experiment('outlier_detection')
if(os.sys.platform != 'win32'):
    ex.observers.append(MongoObserver.create(
        url=collection_url,
        db_name='sacred'))
        
def Glass():
    feature, ground_truth = fetch_uci_glass_outlier()
    feature = scale(feature)
    return (feature, ground_truth)
    
def TPR_TNR(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tpr = cm[1,1]/(cm[1,1] + cm[1,0])
    tnr = cm[0,0]/(cm[0,0] + cm[0,1])
    return (tpr, tnr)

@ex.config
def cfg():
    alg_params = {'_gamma' : 0.1, 'contamination' : 0.04, 'n_neighbors' : 2, 'affinity' : 'laplacian'}
    alg = 'ic' # choices from ['ic', 'lof']
    verbose = True   
 
@ex.automain
def run(alg, alg_params, verbose):
    data, labels = Glass()
    if(alg == 'ic'):
        alg_instance = InfoOutlierDetector(gamma=alg_params['_gamma'],
            affinity=alg_params['affinity'])
    elif(alg == 'lof'):
        alg_instance = LocalOutlierFactor(n_neighbors=alg_params['n_neighbors'], 
            contamination=alg_params['contamination'])
    else:
        raise NameError(alg + ' name not found')
    
    y_predict = alg_instance.fit_predict(data)
    if(alg == 'ic' and verbose):
        print(alg_instance.partition_num_list)
    tpr, tnr = TPR_TNR(labels, y_predict)
    ex.log_scalar("tpr", tpr)        
    ex.log_scalar("tnr", tnr)    
    return (tpr, tnr)    
