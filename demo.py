import argparse
import os
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.covariance import EllipticEnvelope

from info_detection import InfoOutlierDetector
from util import generate_one_blob, generate_two_moon
from util import load_parameters

def get_moon_configuration():
    train_data, _ = generate_two_moon()

    alg = EllipticEnvelope(contamination=0.15)
    ic = InfoOutlierDetector(gamma=0.8)
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

def plot_alg_time(axis, filename, omit_list = ['pdt_r']):
    '''combine different algorithms
    '''
    linestyle_list = ['-','-', '-', '--']
    marker_list = ['o', 'v', 's', '*', '+', 'x', 'D', '1']
    color_list = ['b', 'r', 'g', 'm','y','k','c','#00FF00']
    method_translate = {'pdt_r': 'Kolmogorov', 'dt': 'Narayanan', 'psp_i': 'ours(psp_i)', 'pdt': 'ours(pdt)'}
    f = open(os.path.join('build', filename), 'r')
    data = json.loads(f.read())
    x_data = [int(i) for i in data.keys()]
    one_key = str(x_data[0])
    alg_data = {}
    for i in data[one_key].keys():
        alg_data[i] = []
    for i in data.values():
        for k,v in i.items():
            alg_data[k].append(v)
    index = 0
    for k,v in alg_data.items():
        if(k == 'num_edge' or omit_list.count(k) > 0):
            continue
        axis.plot(x_data, v, label=method_translate[k], linewidth=3, color=color_list[index],
            marker=marker_list[index], markersize=12, linestyle=linestyle_list[index])
        index += 1
    axis.set_ylabel('Time(s)')
    axis.set_xlabel('N(nodes)')    
    axis.xaxis.set_label_coords(1.06, -0.025)
    if filename.find('gaussian') >= 0:
        plot_title = 'Gaussian blob dataset'
    else:
        plot_title = 'Two level graph dataset'
    axis.set_yscale('log')
    axis.set_title(plot_title)
    axis.legend()


def plot_barchart_for_dataset(axis):
    parameter_json = load_parameters()
    dataset_list = ['GaussianBlob', 'Moon', 'Lymphography']
    method_translate = {'ic' : 'ours', 'if' : 'if', 'svm' : 'svm', 'ee' : 'ee', 'lof' : 'lof'}
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
        axis.bar(x - width/2 + offset * width/num_of_algs, v, width/num_of_algs, label = method_translate[k])
        offset += 1
    axis.set_xticks(x)
    axis.set_xticklabels(dataset_list)
    axis.set_ylabel('TNF')
    axis.set_title('Method comparison')
    axis.legend(loc='upper center', bbox_to_anchor=(0.65, 1))

def plot_experimental_results():
    plt.figure(figsize=(18, 5.7))
    plt.subplots_adjust(wspace=.17)
    ax = plt.subplot(1, 3, 1)
    plot_alg_time(ax, '2019-08-26-gaussian.json')
    ax = plt.subplot(1, 3, 2)
    plot_alg_time(ax, '2019-09-19-two_level.json')
    ax = plt.subplot(1, 3, 3)
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
