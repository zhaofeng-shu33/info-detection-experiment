import json
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import scale
from sklearn.datasets import make_blobs, make_moons
try:
    from uci_glass_outlier import fetch_uci_glass_outlier
    from lymphography_outlier import fetch_or_load_lymphography
    from ionosphere_outlier import fetch_or_load_Ionosphere
except:
    pass
BUILD_DIR = 'build'
PARAMETER_FILE = 'parameter.json'

def load_parameters():
    parameter_file_path = os.path.join(BUILD_DIR, PARAMETER_FILE)
    if (os.path.exists(parameter_file_path)):
        json_str = open(parameter_file_path).read()
    else:
        json_str = None            
        with open(parameter_file_path, 'w') as f:
            json_str_new = update_json(json_str)
            f.write(json_str_new)
        print('parameter files written to %s' % PARAMETER_FILE)
    return json.loads(json_str)

def write_parameters(parameter_json):
    parameter_file_path = os.path.join(BUILD_DIR, PARAMETER_FILE)
    with open(parameter_file_path, 'w') as f:
        f.write(json.dumps(parameter_json, indent=4))

def TPR_TNR(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred) # [[tn, fp],[fn, tp]]
    tpr = cm[1,1]/(cm[1,1] + cm[1,0]) # tp / (tp + fn)
    tnr = cm[0,0]/(cm[0,0] + cm[0,1]) # tn / (tn + fp)
    return (tpr, tnr)
    
def Lymphography():
    feature, ground_truth = fetch_or_load_lymphography()
    return (feature, ground_truth)

def Ionosphere():
    feature, ground_truth = fetch_or_load_Ionosphere()
    return (feature, ground_truth)

def Glass():
    feature, ground_truth = fetch_uci_glass_outlier()
    feature = scale(feature)
    return (feature, ground_truth)       
    
def generate_one_blob():
    '''
       generate training data with (300,2)
    '''
    n_samples = 300
    outliers_fraction = 0.15
    n_outliers = int(outliers_fraction * n_samples)
    n_inliers = n_samples - n_outliers
    blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=2)
    data, labels = make_blobs(centers=[[0, 0]], cluster_std=0.5,
               **blobs_params)
    rng = np.random.RandomState(42)
    outlier_data = rng.uniform(low=-6, high=6, size=(n_outliers, 2))
    total_data = np.vstack((data, outlier_data))
    ground_truth = np.ones(n_samples)
    ground_truth[n_inliers:n_samples] = -1
    return (total_data, ground_truth)

def generate_two_moon():
    n_samples = 300
    outliers_fraction = 0.15
    n_outliers = int(outliers_fraction * n_samples)    
    data = 4. * (make_moons(n_samples=n_samples, noise=.05, random_state=0)[0] -
              np.array([0.5, 0.25]))
    rng = np.random.RandomState(34)              
    outlier_data = rng.uniform(low=-6, high=6,
                           size=(n_outliers, 2))
    # make real outlier
    outlier_data[22,1] = 1.5
    outlier_data[30,0] = -1
    outlier_data[23,1] = -4
    outlier_data[31,1] = -3.9
    outlier_data[1,1] = 0.1
    outlier_data[1,0] = -0.1
    outlier_data[10,1] = 2                           
    total_data = np.vstack((data, outlier_data))
    ground_truth = np.ones(n_samples + n_outliers)
    ground_truth[n_samples:(n_samples + n_outliers)] = -1    
    return (total_data, ground_truth)