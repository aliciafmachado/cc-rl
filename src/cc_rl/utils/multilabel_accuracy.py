import numpy as np

def multilabel_accuracy(y_true, y_pred):
    """Calculates accuracy as described here
    https://stackoverflow.com/questions/32239577/getting-the-accuracy-for-multi-label-prediction-in-scikit-learn

    Args:
        y_true (np.array): Ground truth (correct) labels.
        y_pred (np.array): Predicted labels, as returned by a classifier.

    Returns:
        float: Accuracy value, between 0 and 1.
    """
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0] )
        set_pred = set(np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)
