# parametric tuning for info-detection
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize

from util import TPR_TNR
from util import Lymphography
from info_detection import InfoOutlierDetector

SPACE = [
    Real(0.01,0.2, prior='uniform', name='gamma')
]
data, labels = Lymphography()
@use_named_args(SPACE)
def objective(**params):
    global data, labels
    ic = InfoOutlierDetector(affinity='laplacian', **params)
    y_predict = ic.fit_predict(data)
    tpr, tnr = TPR_TNR(labels, y_predict)
    if(tpr < 0.9):
        return 10
    return 0.9 - tpr - tnr
    
if __name__ == '__main__':
    res_gp = gp_minimize(objective, SPACE, n_calls=40, random_state=0)
    print(res_gp)