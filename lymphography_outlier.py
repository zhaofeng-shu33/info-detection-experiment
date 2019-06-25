from os.path import exists, join
from os import makedirs, remove
import tarfile
import arff
import pandas as pd
import numpy as np
from sklearn.datasets.base import RemoteFileMetadata
from sklearn.datasets.base import get_data_home, _fetch_remote

Lymphography = RemoteFileMetadata(
    filename = 'Lymphography.tar.gz',
    url='http://www.dbs.ifi.lmu.de/research/outlier-evaluation/input/Lymphography.tar.gz',
    checksum='1ecb8fc1cc86960bbbe604d8fbf058f01bf4035af1165cc32470f9dced77a8f8'
)
def fetch_or_load_lymphography():
    global Lymphography
    data_home = get_data_home()
    if not exists(data_home):
        makedirs(data_home)
    file_path = join(data_home, 'Lymphography', 'Lymphography_withoutdupl_idf.arff')
    if not exists(file_path):
        data_archive_path = _fetch_remote(Lymphography)
        tf = tarfile.open(data_archive_path)
        tf.extractall(data_home)
        remove(data_path)
    f_descriptor = open(file_path, 'r')
    dataset = arff.load(f_descriptor)
    df = pd.DataFrame(dataset['data'])
    feature = df.iloc[:,1:19].to_numpy()
    ground_truth = np.ones(148)
    for i in [43, 44, 45, 103, 132, 147]:
        ground_truth[i] = -1 
    return (feature, ground_truth)

if __name__ == '__main__':
    fetch_or_load_lymphography() 
