import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.covariance import EllipticEnvelope

from info_detection import InfoOutlierDetector
from util import generate_one_blob, generate_two_moon
    
def get_moon_configuration():
    train_data, _ = generate_two_moon()

    alg = EllipticEnvelope(contamination=0.15)
    ic = InfoOutlierDetector(gamma=0.4)
    return [('Info-Detection', ic, train_data), ('Elliptic Envelope', alg, train_data)]

def get_blob_configuration():
    train_data, _ = generate_one_blob()

    ic = InfoOutlierDetector(gamma=0.5) # 1/num_of_features
    return [('Info-Detection', ic, train_data)]  
    
def plot_common_routine(combination, suffix):
    xx, yy = np.meshgrid(np.linspace(-7, 7, 150),
                     np.linspace(-7, 7, 150))
    num_of_alg = len(combination)                     
    plt.figure(figsize=(6*num_of_alg, 6))
    for i, combination_tuple in enumerate(combination):
        plt.subplot(1, num_of_alg, i+1)
        alg_name, alg_class, train_data = combination_tuple
        y_pred = alg_class.fit_predict(train_data)
        Z = alg_class.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)    
        plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
        plt.scatter(train_data[y_pred==1,0], train_data[y_pred==1,1], s=5)
        plt.scatter(train_data[y_pred==-1,0], train_data[y_pred==-1,1], s=5)
        plt.title(alg_name, fontsize=20)
    plt.tight_layout()  
    if not(os.path.exists('build')):
        os.mkdir('build')    
    plt.savefig('build/outlier_boundary_illustration.' + suffix) 
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='all', choices=['all', 'blob', 'moon'])
    parser.add_argument('--figure_suffix', default='eps', choices=['eps', 'pdf', 'svg'])    
    args = parser.parse_args()
    combination_list = []
    if(args.dataset == 'all' or args.dataset == 'blob'):
        combination_list.extend(get_blob_configuration())
    if(args.dataset == 'all' or args.dataset == 'moon'):
        combination_list.extend(get_moon_configuration())
    plot_common_routine(combination_list, args.figure_suffix)

