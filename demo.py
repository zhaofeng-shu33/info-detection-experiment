import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.covariance import EllipticEnvelope

from info_detection import InfoOutlierDetector
from util import generate_one_blob, generate_two_moon
    
def run_moon():
    train_data, _ = generate_two_moon()

    alg = EllipticEnvelope(contamination=0.15)
    ic = InfoOutlierDetector(gamma=0.4)
    plot_common_routine([('Elliptic Envelope', alg), ('Info-detection', ic)], train_data, 'moon')    

def run_blob():
    train_data, _ = generate_one_blob()

    alg = EllipticEnvelope(contamination=0.15)
    ic = InfoOutlierDetector(gamma=0.5) # 1/num_of_features
    plot_common_routine([('Elliptic Envelope', alg), ('Info-detection', ic)], train_data, 'blob')        
    
def plot_common_routine(alg_list, train_data, data_name):
    xx, yy = np.meshgrid(np.linspace(-7, 7, 150),
                     np.linspace(-7, 7, 150))
    num_of_alg = len(alg_list)                     
    plt.figure(figsize=(6*num_of_alg, 6))
    for i, alg_tuple in enumerate(alg_list):
        plt.subplot(1, num_of_alg, i+1)
        alg_name, alg_class = alg_tuple
        y_pred = alg_class.fit_predict(train_data)
        Z = alg_class.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)    
        plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
        plt.scatter(train_data[y_pred==1,0], train_data[y_pred==1,1], s=5)
        plt.scatter(train_data[y_pred==-1,0], train_data[y_pred==-1,1], s=5)
        plt.title(alg_name)
    plt.savefig('build/outlier_compare_%s.eps' % data_name) 
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='all', choices=['all', 'blob', 'moon'])
    args = parser.parse_args()
    if(args.dataset == 'all' or args.dataset == 'blob'):
        run_blob()
    if(args.dataset == 'all' or args.dataset == 'moon'):
        run_moon()
    

