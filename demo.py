import argparse
import os
import json
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.covariance import EllipticEnvelope

from info_detection import InfoOutlierDetector
from util import generate_one_blob, generate_two_moon
from util import load_parameters

class FourPart:
    def __init__(self, _np, _gamma=1):
        '''
        np is the number of points at each part
        '''
        #  (0, 0.1) to (0, 0.2)
        self._gamma = _gamma
        pos_list = []
        part_center = [[3,3],[3,-3],[-3,-3],[-3,3]]
        rng = np.random.RandomState(42)
        for i in range(4): # radius: 0.1*i
            for j in range(_np):
                x = part_center[i][0] + rng.normal(0,1) # standard normal distribution disturbance
                y = part_center[i][1] + rng.normal(0,1)                
                pos_list.append([x, y])
        self.pos_list = np.asarray(pos_list)

def get_moon_configuration(compare_elliptic=True):
    train_data, _ = generate_two_moon()

    ic = InfoOutlierDetector(gamma=0.8)
    if compare_elliptic:
        alg = EllipticEnvelope(contamination=0.15)
        return [('Info-Detection on Moon', ic, train_data), ('Elliptic Envelope on Moon', alg, train_data)]
    return [('Info-Detection on Moon', ic, train_data)]

def get_blob_configuration():
    train_data, _ = generate_one_blob()

    ic = InfoOutlierDetector(gamma=0.5) # 1/num_of_features
    return [('Info-Detection on GaussianBlob', ic, train_data)]

def get_4_blobs_configuration():
    four_part = FourPart(25)
    train_data = np.vstack((four_part.pos_list, np.array([0, 0])))
    ic = InfoOutlierDetector(gamma=0.5) # 1/num_of_features
    return [('Info-Detection on 4-GaussianBlobs', ic, train_data)]    

def plot_common_routine(show_pic, combination, suffix):
    xx, yy = np.meshgrid(np.linspace(-7, 7, 150),
                     np.linspace(-7, 7, 150))
    num_of_alg = len(combination)                     
    plt.figure(figsize=(3 * num_of_alg, 3))
    label_text = ['(a)', '(b)', '(c)', '(d)']
    for i, combination_tuple in enumerate(combination):
        plt.subplot(1, num_of_alg, i+1)
        alg_name, alg_class, train_data = combination_tuple
        y_pred = alg_class.fit_predict(train_data)
        Z = alg_class.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)    
        plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
        plt.scatter(train_data[y_pred>=1,0], train_data[y_pred>=1,1], s=5)
        plt.scatter(train_data[y_pred==-1,0], train_data[y_pred==-1,1], s=5)
        plt.xlabel(label_text[i])
        plt.title(alg_name)
    plt.tight_layout()  
    if not(os.path.exists('build')):
        os.mkdir('build')    
    plt.savefig('build/outlier_boundary_illustration.' + suffix, transparent=True)
    if show_pic:
        plt.show()

def plot_alg_time(axis, filename, omit_list = ['pdt'], show_labels=True):
    '''combine different algorithms
    '''
    linestyle_list = ['-','-', '-', '--']
    marker_list = ['o', 'v', 's', '*', '+', 'x', 'D', '1']
    color_list = ['b', 'r', 'g', 'm','y','k','c','#00FF00']
    method_translate = {'pdt_r': 'Kolmogorov', 'dt': 'Narayanan', 'psp_i': 'Ours', 'pdt': 'ours(pdt)'}
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
    axis.xaxis.set_ticks(x_data)
    if show_labels:
        axis.set_ylabel('Time(s)', fontsize=16)
        axis.set_xlabel('N(Nodes)')    
        axis.xaxis.set_label_coords(-0.14, -0.035)
    if filename.find('gaussian') >= 0:
        plot_title = '(a) GaussianBlobs'
    else:
        plot_title = '(b) Two-level graph'
    axis.set_yscale('log')
    axis.set_title(plot_title)
    axis.legend().set_zorder(1)


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
        axis.bar(x - width/2 + offset * width/num_of_algs, v, width/num_of_algs, label=method_translate[k])
        offset += 1
    axis.set_xticks(x)
    axis.set_xticklabels(dataset_list)
    axis.set_ylabel('TNR', fontsize=16)
    axis.set_title('(c) Method comparison')
    axis.legend(loc='upper center', bbox_to_anchor=(0.68, 1)).set_zorder(0)

def get_file_path(keyword):
    match_file_name = ''
    for i in os.listdir('build'):
        if i.find(keyword) > 0:
            match_file_name = i
            break
    if match_file_name == '':
        raise FileNotFoundError(keyword)
    return match_file_name

def plot_experimental_results(show_pic, suffix):
    _, (a1, a2, a3) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [3.2,3,4]}, figsize=(10.5, 3))
    plt.subplots_adjust(wspace=0, right=1, left=0)
    # speed comparison data file is available at https://github.com/zhaofeng-shu33/pspartition-speed-compare
    plot_alg_time(a1, get_file_path('gaussian'))
    plot_alg_time(a2, get_file_path('two_level'), show_labels=False)
    plot_barchart_for_dataset(a3)
    plt.tight_layout()
    plt.savefig('build/experimental_results_triple.' + suffix, transparent=True)
    if show_pic:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='boundary_plot', choices=['boundary_plot', 'experiment_matrix_plot'])
    parser.add_argument('--dataset', default='all', choices=['all', 'blob', 'moon', '4-blobs'])
    parser.add_argument('--figure_suffix', default='eps', choices=['eps', 'pdf', 'svg', 'png'])    
    parser.add_argument('--show_pic', default=False, type=bool, nargs='?', const=True)
    parser.add_argument('--omit_elliptic', default=False, type=bool, const=True, nargs='?')
    args = parser.parse_args()
    if args.task == 'boundary_plot':
        combination_list = []
        if(args.dataset == 'all' or args.dataset == 'blob'):
            combination_list.extend(get_blob_configuration())
        if(args.dataset == 'all' or args.dataset == '4-blobs'):
            combination_list.extend(get_4_blobs_configuration())
        if(args.dataset == 'all' or args.dataset == 'moon'):
            combination_list.extend(get_moon_configuration(not args.omit_elliptic))
        plot_common_routine(args.show_pic, combination_list, args.figure_suffix)
    else:
        plot_experimental_results(args.show_pic, args.figure_suffix)
