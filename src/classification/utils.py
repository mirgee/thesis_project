from functools import wraps
from sklearn import metrics


def print_params(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        print(f'{args} {kwargs}')
        res = f(*args, **kwargs)
        return res
    return wrapper


def get_cm(cm):
    s = '$\\left( \\begin{smallmatrix} '
    for i, r in enumerate(cm):
        s += ' & '.join(map(str, r)) + ' \\\\ ' if i < len(cm)-1 else ' & '.join(map(str, r))
    s += ' \\end{smallmatrix} \\right)$'
    return s


def scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    # return metrics.accuracy_score(y, y_pred)
    # return (metrics.f1_score(y, y_pred, average='weighted', labels=np.unique(y_pred)))
            # + metrics.precision_score(y, y_pred, average='weighted') 
            # + metrics.recall_score(y, y_pred, average='weighted') 
            # + metrics.accuracy_score(y, y_pred))
    return metrics.roc_auc_score(y, y_pred, average='weighted')
