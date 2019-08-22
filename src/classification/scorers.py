

def scorer_factory(metric, **kwargs):
    def scorer(estimator, X, y):
        y_pred = estimator.predict(X)
        return metric(y, y_pred, **kwargs)
    return scorer
