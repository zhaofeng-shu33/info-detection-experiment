from sklearn.datasets import make_blobs, make_moons
import matplotlib.pyplot as plt
import numpy as np
from info_detection import InfoOutlierDetector
from sklearn.covariance import EllipticEnvelope
import argparse
def generate_one_blob():
    '''
       generate training data with (300,2)
    '''
    n_samples = 300
    outliers_fraction = 0.15
    n_outliers = int(outliers_fraction * n_samples)
    n_inliers = n_samples - n_outliers
    blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=2)
    data, labels = make_blobs(centers=[[0, 0], [0, 0]], cluster_std=0.5,
               **blobs_params)
    rng = np.random.RandomState(42)
    outlier_data = rng.uniform(low=-6, high=6, size=(n_outliers, 2))
    total_data = np.vstack((data, outlier_data))
    return total_data

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
    return total_data
    
def run_moon():
    train_data = generate_two_moon()

    alg = EllipticEnvelope(contamination=0.15)
    ic = InfoOutlierDetector(gamma=0.4)
    plot_common_routine([('Elliptic Envelope', alg), ('Info-detection', ic)], train_data, 'moon')    

def run_blob():
    train_data = generate_one_blob()

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
    

