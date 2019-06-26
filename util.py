from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import scale

from uci_glass_outlier import fetch_uci_glass_outlier
from lymphography_outlier import fetch_or_load_lymphography

def TPR_TNR(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tpr = cm[1,1]/(cm[1,1] + cm[1,0])
    tnr = cm[0,0]/(cm[0,0] + cm[0,1])
    return (tpr, tnr)
    
def Lymphography():
    feature, ground_truth = fetch_or_load_lymphography()
    return (feature, ground_truth)
        
def Glass():
    feature, ground_truth = fetch_uci_glass_outlier()
    feature = scale(feature)
    return (feature, ground_truth)       