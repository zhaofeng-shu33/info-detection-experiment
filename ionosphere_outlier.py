from os.path import exists, join
from os import makedirs, remove
import tarfile
import arff
import pandas as pd
import numpy as np
from sklearn.datasets.base import RemoteFileMetadata
from sklearn.datasets.base import get_data_home, _fetch_remote

Ionosphere = RemoteFileMetadata(
    filename = 'Ionosphere.tar.gz',
    url='https://www.dbs.ifi.lmu.de/research/outlier-evaluation/input/Ionosphere.tar.gz',
    checksum='6e4b224f6270b2626fe4fc9f888f1c0f317c4642b241ebb32fb534cc97b2b0da'
)
def fetch_or_load_Ionosphere():
    global Ionosphere
    data_home = get_data_home()
    if not exists(data_home):
        makedirs(data_home)
    file_path = join(data_home, 'Ionosphere', 'Ionosphere_withoutdupl_norm.arff')
    if not exists(file_path):
        data_archive_path = _fetch_remote(Ionosphere)
        tf = tarfile.open(data_archive_path)
        tf.extractall(data_home)
        tf.close()
        remove(data_archive_path)
    f_descriptor = open(file_path, 'r')
    dataset = arff.load(f_descriptor)
    df = pd.DataFrame(dataset['data'])
    feature = df.iloc[:,0:32].to_numpy()
    ground_truth = df[33].astype('category').cat.codes.to_numpy()
    ground_truth = 1 - 2 * ground_truth
    return (feature, ground_truth)

if __name__ == '__main__':
    fetch_or_load_Ionosphere() 
