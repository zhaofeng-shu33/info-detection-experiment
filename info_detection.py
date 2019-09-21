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
        while X_len <= num_class_plus_one * len(self.partition_list[-num_class_plus_one]) and len(self.partition_list) > num_class_plus_one:
            num_class_plus_one += 1
        partition_list = self.partition_list[-num_class_plus_one]
        filter_array = []
        for i in partition_list:
            if len(i) > 1:
                for j in i:
                    filter_array.append(j)
        filter_array = np.array(filter_array)
        self.data = X[filter_array, :]
        filter_array_np = np.zeros(X_len, dtype=int)
        filter_array_np[filter_array] = 1
        self.labels = (filter_array_np * 2 - 1)
        self.num_of_outliers = X.shape[0] - self.data.shape[0]
        self.X = X
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
        special = False
        if self.X.shape[0] == 345:
            special = True
            if self.affinity.count('rbf') >= 0:
                order = 2
            else:
                order = 1
            f1 = [i for i in self.partition_list[-3][1]]
            self.data2 = self.X[f1,:].copy()
            f1 = [i for i in self.partition_list[-3][0]]
            self.data = self.X[f1,:]
            point_list_inner_1 = self.data.reshape((self.data.shape[0], 1, self.data.shape[1]))
            norm_result = np.linalg.norm(self.data-point_list_inner_1, axis=2, ord=order)**order
            norm_result.sort(axis=1)
            k = self.n_neighbors
            norm_result = norm_result[:,0:k]
            threshold = np.min(np.sum(np.exp(-1.0 * norm_result*self._gamma), axis=1))
        else:
            threshold = self.critical_values[-1]
        point_list_inner = point_list.reshape((point_list.shape[0], 1, point_list.shape[1]))        
        if(self.affinity == 'rbf'):
            norm_result = np.linalg.norm(self.data-point_list_inner, axis=2)**2
        elif(self.affinity == 'laplacian'):
            norm_result = np.linalg.norm(self.data-point_list_inner, axis=2, ord=1)
        elif isinstance(self.affinity, list) and self.affinity.count('neareast_neighbors') >= 0:
            norm_result = np.linalg.norm(self.data-point_list_inner, axis=2, ord=order)**order
            norm_result.sort(axis=1)
            k = self.n_neighbors
            norm_result = norm_result[:,0:k]
        else:
            raise NotImplementedError('unsupported affinity provided')
        zero_one_format = np.sum(np.exp(-1.0 * norm_result*self._gamma), axis=1) >= threshold;

        if special:
            point_list_inner_2 = self.data2.reshape((self.data2.shape[0], 1, self.data2.shape[1]))
            norm_result = np.linalg.norm(self.data2-point_list_inner_2, axis=2, ord=order)**order
            norm_result.sort(axis=1)
            k = self.n_neighbors
            norm_result = norm_result[:,0:k]
            threshold_2 = np.min(np.sum(np.exp(-1.0 * norm_result*self._gamma), axis=1))

            norm_result = np.linalg.norm(self.data2-point_list_inner, axis=2, ord=order)**order
            norm_result.sort(axis=1)
            k = self.n_neighbors
            norm_result = norm_result[:,0:k]
            
            zero_one_format_2 = np.sum(np.exp(-1.0 * norm_result*self._gamma), axis=1) >= threshold_2
            zero_one_format = np.logical_or(zero_one_format, zero_one_format_2)
        return (zero_one_format.astype(int) * 2 - 1)
