# outlier detecter based on info_clustering
from info_cluster import InfoCluster
import numpy as np
def statistic_get(int_list):
    dic = {}
    for i in int_list:
        if(dic.get(i)):
            dic[i] += 1
        else:
            dic[i] = 1
    return dic
class InfoOutlierDetector(InfoCluster):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X):
        super().fit(X)
        X_len = X.shape[0]
        num_class_plus_one = 2
        # outlier detection assuming there are clustering structures in inliers
        # https://github.com/zhaofeng-shu33/info-clustering-research/blob/23f8445cebcd56f325635e404c4f386652887f46/clustering_with_outliers.tex
        while X_len <= num_class_plus_one * len(self.partition_list[-num_class_plus_one]) and len(self.partition_list) > num_class_plus_one:
            num_class_plus_one += 1
        partition_list = self.partition_list[-num_class_plus_one]
        filter_array = []
        self.num_of_outliers = 0
        for i in partition_list:
            if len(i) > 1:
                    filter_array.append(list(i))
            else:
                self.num_of_outliers += 1
        self.num_of_class = num_class_plus_one - 1
        filter_array = np.array(filter_array)
        self.data = []
        for i in filter_array:
            self.data.append(X[i, :])
        self.labels = np.zeros(X_len, dtype=int) - 1 # outliers labeled to 0
        cnt = 1
        for i in filter_array:
            self.labels[i] = cnt
            cnt += 1

    def fit_predict(self, X):
        self.fit(X)
        return self.labels

    def predict(self, point_list):
        '''predict whether the new data is outlier or not

        Parameters
        ----------
        point_list: array-like, shape (n_samples, n_features)

        Returns
        -------
        prediction_list: array-like shape (n_samples,), dtype=bool
        '''
        threshold = self.critical_values[-self.num_of_class]
        zero_one_format = np.zeros(point_list.shape[0], dtype=bool)
        point_list_inner = point_list.reshape((point_list.shape[0], 1, point_list.shape[1]))        
        for _data in self.data:
            if(self.affinity == 'rbf'):
                norm_result = np.linalg.norm(_data - point_list_inner, axis=2)**2
            elif(self.affinity == 'laplacian'):
                norm_result = np.linalg.norm(_data - point_list_inner, axis=2, ord=1)
            elif isinstance(self.affinity, list) and self.affinity.count('neareast_neighbors') >= 0 and self.affinity.count('rbf') >= 0:
                norm_result = np.linalg.norm(_data - point_list_inner, axis=2)**2
                norm_result.sort(axis=1)
                k = self.n_neighbors
                norm_result = norm_result[:,0:k]
            else:
                raise NotImplementedError('unsupported affinity provided')
            zero_one_format = np.logical_or(zero_one_format, np.sum(np.exp(-1.0 * norm_result * self._gamma), axis=1) >= threshold)
        return (zero_one_format.astype(int) * 2 - 1)
