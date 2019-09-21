import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.covariance import EllipticEnvelope

from info_detection import InfoOutlierDetector
from util import generate_one_blob, generate_two_moon
from util import load_parameters

def get_moon_configuration():
    train_data, _ = generate_two_moon()

    alg = EllipticEnvelope(contamination=0.15)
    ic = InfoOutlierDetector(gamma=0.41, n_neighbors=20, affinity=['rbf', 'nearest_neighbors'])
    return [('Info-Detection on Moon', ic, train_data), ('Elliptic Envelope on Moon', alg, train_data)]

def get_blob_configuration():
    train_data, _ = generate_one_blob()

    ic = InfoOutlierDetector(gamma=0.5) # 1/num_of_features
    return [('Info-Detection on GaussianBlob', ic, train_data)]  
    
def plot_common_routine(combination, suffix):
    xx, yy = np.meshgrid(np.linspace(-7, 7, 150),
                     np.linspace(-7, 7, 150))
    num_of_alg = len(combination)                     
    plt.figure(figsize=(6 * num_of_alg, 6))
    label_text = ['(a)', '(b)', '(c)']
    for i, combination_tuple in enumerate(combination):
        plt.subplot(1, num_of_alg, i+1)
        alg_name, alg_class, train_data = combination_tuple
        y_pred = alg_class.fit_predict(train_data)
        Z = alg_class.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)    
        plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
        plt.scatter(train_data[y_pred==1,0], train_data[y_pred==1,1], s=5)
        plt.scatter(train_data[y_pred==-1,0], train_data[y_pred==-1,1], s=5)
        plt.xlabel(label_text[i], fontsize=20)
        plt.title(alg_name, fontsize=20)
    plt.tight_layout()  
    if not(os.path.exists('build')):
        os.mkdir('build')    
    plt.savefig('build/outlier_boundary_illustration.' + suffix) 

def plot_barchart_for_dataset(axis):
    parameter_json = load_parameters()
    dataset_list = ['GaussianBlob', 'Moon', 'Lymphography']
    alg_dic = {}
    for i in parameter_json['GaussianBlob'].keys():
        alg_dic[i] = []
    for dataset in dataset_list:
        for i,v in parameter_json[dataset].items():
            alg_dic[i].append(v['tnr'])
    x = np.arange(len(dataset_list))  # the label locations
    width = 0.35  # the width of the bars
    num_of_algs = len(alg_dic.keys())
    offset = 0
    for k, v in alg_dic.items():
        axis.bar(x - width/2 + offset * width/num_of_algs, v, width/num_of_algs, label = k)
        offset += 1
    axis.set_xticks(x)
    axis.set_xticklabels(dataset_list)
    axis.set_ylabel('TNF')
    axis.legend()

def plot_experimental_results():
    fig, ax = plt.subplots()
    plot_barchart_for_dataset(ax)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='boundary_plot', choices=['bounadry_plot', 'experiment_matrix_plot'])
    parser.add_argument('--dataset', default='all', choices=['all', 'blob', 'moon'])
    parser.add_argument('--figure_suffix', default='eps', choices=['eps', 'pdf', 'svg'])    
    args = parser.parse_args()
    if args.task == 'boundary_plot':
        combination_list = []
        if(args.dataset == 'all' or args.dataset == 'blob'):
            combination_list.extend(get_blob_configuration())
        if(args.dataset == 'all' or args.dataset == 'moon'):
            combination_list.extend(get_moon_configuration())
        plot_common_routine(combination_list, args.figure_suffix)
    else:
        plot_experimental_results()
