import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, train_test_split

from classification.utils import print_params
from sklearn import metrics

from config import CHANNEL_NAMES
from data.utils import get_metapkl, prepare_dfs


@print_params
def predict(lab, ba, cols, estimator, gs=None, evaluate_on_all=False,
            channels=CHANNEL_NAMES, selector=None,
            print_incorrectly_predicted=False, show_selected=False, seed=213,
            eval_cv=True):
    df, df_bef, df_aft = prepare_dfs('all')
    metapkl = get_metapkl()
    if cols is None:
        df = df.loc[(slice(None), slice(ba)), channels]
    else:
        df = df.loc[(slice(None), slice(ba)), (channels, (cols))]
    X = df.dropna()
    X.columns = X.columns.droplevel(0)
    y = X.join(metapkl)[lab]
    X = X[y.isin([-1, 1])]
    y = y[y.isin([-1, 1])]

    if selector is not None:
        X = selector.fit_transform(X, y)
        if show_selected and hasattr(selector, 'get_support'):
            # print('(\'' + '\', \''.join(np.unique(df.columns.values[selector.get_support()])) + '\')')
            print(list(df.columns.values[selector.get_support()]))

    if gs is not None:
        gs = gs.fit(X, y)
        estimator = gs.best_estimator_

    if eval_cv:
        unique, counts = np.unique(y, return_counts=True)
        print('Class distribution: ', dict(zip(unique, counts)))
        scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        scores = cross_validate(estimator, X, y, cv=5, scoring=scoring, return_train_score=False)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores['test_'+scoring[0]].mean(), scores['test_'+scoring[0]].std()))
        print("Precision: %0.2f (+/- %0.2f)" % (scores['test_'+scoring[1]].mean(), scores['test_'+scoring[1]].std()))
        print("Recall: %0.2f (+/- %0.2f)" % (scores['test_'+scoring[2]].mean(), scores['test_'+scoring[2]].std()))
        print("F1: %0.2f (+/- %0.2f)" % (scores['test_'+scoring[3]].mean(), scores['test_'+scoring[3]].std()))
        print(
            '{: <3.2f} $\pm$ {:<3.2f} & {: <3.2f} $\pm$ {: <3.2f} & '.format(scores['test_'+scoring[0]].mean(), scores['test_'+scoring[0]].std(), 
                                                                            scores['test_'+scoring[1]].mean(), scores['test_'+scoring[1]].std()) +
            '{: <3.2f} $\pm$ {:<3.2f} & {: <3.2f} $\pm$ {: <3.2f} & '.format(scores['test_'+scoring[2]].mean(), scores['test_'+scoring[2]].std(),
                                                                            scores['test_'+scoring[3]].mean(), scores['test_'+scoring[3]].std()) +
            ' \\\\ \hline'
        )
    else:
        X_train, X_test, y_train, y_test = \
            train_test_split(
                X, y, test_size=0.3, random_state=seed)

        unique, counts = np.unique(y_train, return_counts=True)
        print('Training distribution: ', dict(zip(unique, counts)))
        unique, counts = np.unique(y_test, return_counts=True)
        print('Testing distribution: ', dict(zip(unique, counts)))
        y_train = y_train.astype('int')
        estimator = estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        y_pred = y_pred.astype('int')
        y_test = y_test.astype('int')
        print("Accuracy score: %.2f" % metrics.accuracy_score(y_test, y_pred))
        print('Confusion matrix:\n', metrics.confusion_matrix(y_test, y_pred))
        print('Precision score: ', metrics.precision_score(y_test, y_pred, average='weighted'))
        print('Recall score: ', metrics.recall_score(y_test, y_pred, average='weighted'))
        print('f1 score: ', metrics.f2_score(y_test, y_pred, average='weighted'))
        print('ROC AUC score: ', metrics.roc_auc_score(y_test, y_pred, average='weighted'))
        # print('Coefficients: \n', np.array(X.stack().columns)[estimator.support_])
        # print(
        #     '{:.2f} & {:.2f} & {:.2f} & {} & {} \\\\ \hline'.format(
        #         metrics.accuracy_score(y_test, y_pred),
        #         metrics.f1_score(y_test, y_pred, average='weighted'),
        #         metrics.roc_auc_score(y_test, y_pred, average='weighted'),
        #         get_cm(metrics.confusion_matrix(y_test, y_pred)),
        #         ', '.join(channels)
        #     ))
    
    if print_incorrectly_predicted:
        print('Incorrectly predicted:')
        print(pd.DataFrame(y_test[y_pred != y_test]).join(X))
    
    if evaluate_on_all:
        y_pred = estimator.predict(X)
        y_pred = y_pred.astype('int')

        print("Accuracy score: %.2f" % metrics.accuracy_score(y, y_pred))
        print('Confusion matrix:\n', metrics.confusion_matrix(y, y_pred))
        print('Precision score: ', metrics.precision_score(y, y_pred, average='weighted'))
        print('Recall score: ', metrics.recall_score(y, y_pred, average='weighted'))
        print('f1 score: ', metrics.f1_score(y, y_pred, average='weighted'))
        print('ROC AUC score: ', metrics.roc_auc_score(y, y_pred, average='weighted'))
        if print_incorrectly_predicted:
            print('Incorrectly predicted:')
            print(pd.DataFrame(y[y_pred != y]).join(X))
    return estimator
