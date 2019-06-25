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
        super().fit(X, use_psp_i=True)
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
    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)
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
        if(self.affinity=='rbf'):
            power_coeffient = 2
        elif(self.affinity=='laplacian'):
            power_coeffient = 1
        point_list_inner = point_list.reshape((point_list.shape[0], 1, point_list.shape[1]))
        zero_one_format = np.sum(np.exp(-1.0 * np.linalg.norm(self.data-point_list_inner, axis=2)**power_coeffient*self._gamma), axis=1) >= threshold;
        return (zero_one_format.astype(int) * 2 - 1)
