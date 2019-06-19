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
    def __init__(self, gamma=1):
        # currently only rbf metric is supported
        super().__init__(gamma=gamma)
    def fit(self, X, **kwargs):
        super().fit(X, **kwargs)
        predict_cat = self.partition_num_list[-2]
        labels = np.asarray(self.get_category(predict_cat))
        dic = statistic_get(labels)
        label_max = -1
        for k,v in dic.items():
            if(v > label_max):
                label_max = v
                label_num = k
        # save the training data for prediction use        
        self.data = X[labels == label_num, :]
    def predict(self, point_list):
        '''predict whether the new data is outlier or not
        
        Parameters
        ----------
        point_list: array-like, shape (n_samples, n_features)

        Returns
        -------
        prediction_list: array-like shape (n_samples,), dtype=bool
        '''
        threshold = self.critical_values[-1]
        point_list_inner = point_list.reshape((point_list.shape[0], 1, point_list.shape[1]))
        return np.sum(np.exp(-1.0 * np.linalg.norm(self.data-point_list_inner, axis=2)*self._gamma), axis=1) >= threshold;
