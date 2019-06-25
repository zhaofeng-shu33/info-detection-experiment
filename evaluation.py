import pdb

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import scale
from sklearn.neighbors import LocalOutlierFactor
from sacred import Experiment

from info_detection import InfoOutlierDetector
from uci_glass_outlier import fetch_uci_glass_outlier
ex = Experiment('test_exp')

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
    alg_params = {'_gamma' : 0.2, 'contamination' : 0.04, 'n_neighbors' : 2}
    alg = 'ic' # choices from ['ic', 'lof']
    
@ex.automain
def run(alg, alg_params):
    data, labels = Glass()
    if(alg == 'ic'):
        alg_instance = InfoOutlierDetector(gamma=alg_params['_gamma'])
    elif(alg == 'lof'):
        alg_instance = LocalOutlierFactor(n_neighbors=alg_params['n_neighbors'], 
            contamination=alg_params['contamination'])
    else:
        raise NameError(alg + ' name not found')
    
    y_predict = alg_instance.fit_predict(data)
    tpr, tnr = TPR_TNR(labels, y_predict)
    ex.log_scalar("tpr", tpr)        
    ex.log_scalar("tnr", tnr)    
    print(tpr, tnr)    
