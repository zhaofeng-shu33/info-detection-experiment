# parametric tuning for info-detection
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize

from util import TPR_TNR
from util import generate_two_moon
from info_detection import InfoOutlierDetector

SPACE = [Integer(2, 50, name='n_neighbors'), 
    Real(0.01,0.6, prior='uniform', name='gamma')
]
data, labels = generate_two_moon()
@use_named_args(SPACE)
def objective(**params):
    global data, labels
    ic = InfoOutlierDetector(affinity=['rbf','nearest_neighbors'], **params)
    y_predict = ic.fit_predict(data)
    tpr, tnr = TPR_TNR(labels, y_predict)
    if(tpr < 0.9):
        return 10
    return 0.9 - tpr - tnr
    
if __name__ == '__main__':
    res_gp = gp_minimize(objective, SPACE, n_calls=20, random_state=0)
    print(res_gp)