# outlier detecter based on info_clustering
from info_cluster import InfoCluster
import numpy as np
class InfoOutlierDetector(InfoCluster):
    def __init__(self, gamma=1):
        # currently only rbf metric is supported
        super().__init__(gamma=gamma)
    def fit(self, X, **kwargs):
        # save the training data for prediction use
        self.data = X
        super().fit(X, **kwargs)
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
