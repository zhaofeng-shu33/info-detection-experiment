from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from info_detection import InfoOutlierDetector
from sklearn.covariance import EllipticEnvelope

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

if __name__ == '__main__':
    train_data = generate_one_blob()
    xx, yy = np.meshgrid(np.linspace(-7, 7, 150),
                     np.linspace(-7, 7, 150))

    alg = EllipticEnvelope(contamination=0.15)
    y_pred = alg.fit_predict(train_data)
    # for y_pred, -1 represents outlier, 1 represents normal point
    Z = alg.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ic = InfoOutlierDetector(gamma=0.5) # 1/num_of_features
    ic.fit(train_data, use_psp_i=True) # use the fastest implementation
    predict_cat = ic.partition_num_list[-2]
    labels = np.asarray(ic.get_category(predict_cat))
    Z1 = ic.predict(np.c_[xx.ravel(), yy.ravel()])
    Z1 = Z1.reshape(xx.shape)

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
    plt.scatter(train_data[y_pred==1,0], train_data[y_pred==1,1],s=5)
    plt.scatter(train_data[y_pred==-1,0], train_data[y_pred==-1,1],s=5)
    plt.title('Elliptic Envelope')
    plt.subplot(1,2,2)
    plt.contour(xx, yy, Z1, levels=[0.5], linewidths=2, colors='black')
    plt.scatter(train_data[labels==0,0], train_data[labels==0,1],s=5)
    plt.scatter(train_data[labels>0,0], train_data[labels>0,1],s=5)
    plt.title('info-detection')
    plt.savefig('build/outlier_compare.eps')

