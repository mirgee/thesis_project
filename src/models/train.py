import logging
import os
from random import randint

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, metrics, svm
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

import statsmodels.api as sm
from config import LABELED_ROOT, VISUAL_ROOT

TEST_SIZE = 0.3


def summary(X, y):
    logit_model = sm.Logit(y, X)
    result = logit_model.fit()
    summary = result.summary2()
    logging.info(f'Summary for logistic regression classifier: \n{summary}')


def evaluate(model, X_test, y_test):
    logging.info('Evaluating classifier...')

    y_pred = model.predict(X_test)
    logging.info('Accuracy on the test set: {:.2f}'.format(model.score(
        X_test, y_test)))

    cmatrix = confusion_matrix(y_test, y_pred)
    logging.info(f'Confusion matrix: \n{cmatrix}')

    report = classification_report(y_test, y_pred)
    logging.info(f'Classification report: \n{report}')


def plot_roc_curve(clf, X_test, y_test):
    logit_roc_auc = roc_auc_score(y_test, clf.predict(X_test))
    fpr, tpr, thresholds = roc_curve(
        y_test, clf.predict_proba(X_test)[:, 1])

    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(VISUAL_ROOT, 'Log_ROC'))

    plt.show()


def select_from_model(n, clf, X_train, y_train):
    logging.info('Selecting features...')

    sfm = SelectFromModel(clf, threshold=0.25)
    sfm.fit(X_train, y_train)
    n_features = sfm.transform(X_train).shape[1]
    while n_features > n:
        sfm.threshold += 0.1
        X_transform = sfm.transform(X_train)
        n_features = X_transform.shape[1]
    return X_transform, sfm.get_support()


def select_rfe(n, clf, X_train, y_train):
    rfe = RFE(clf, n)
    fit = rfe.fit(X_train, y_train)
    X_transform = rfe.transform(X_train)

    return X_transform, rfe.get_support()


def train(clf_name):
    logging.info('Performing logistic regression on the dataset...')

    df_X = pd.read_pickle(os.path.join(LABELED_ROOT, 'splits.pickle'))
    # Feature selection
    # rows = [row for row in df_X.index if row[0].startswith('b')]
    # Only total param
    df_X = df_X.loc[:, (slice(None), [4], slice(None))].unstack().dropna()

    df_y = pd.read_pickle(os.path.join(LABELED_ROOT, 'labels_depressed.pickle'))
    df = df_X.join(df_y)

    logging.info(f'The dataframe used for training: \n{df}')

    X = df.loc[:, df.columns != 'label'].values
    y = df.loc[:, 'label'].values
    # y = y.astype('int')

    # summary(X, y)

    X_train, X_test, y_train, y_test = \
        train_test_split(
            X, y, test_size=TEST_SIZE, random_state=randint(0, 1000))
    if clf_name == 'logistic-regression':
        clf = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
        X_train, supp = select_from_model(15, clf, X_train, y_train)
        X_test = X_test[:, supp]

        logging.info('Selected features: {}'.format(
            [x for x, s in zip(list(df.columns), supp) if s]))
    elif clf_name == 'svm':
        # clf = svm.SVC(kernel='rbf', gamma='scale')
        clf = svm.SVC(kernel='rbf', probability=True)
    else:
        raise NotImplemented(f'Classifier {clf} not implemented.')

    model = clf.fit(X_train, y_train)

    evaluate(model, X_test, y_test)

    plot_roc_curve(clf, X_test, y_test)


@click.command()
@click.argument('clf_name', default='logistic-regression')
def main(clf_name, input_path=LABELED_ROOT):
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    train(clf_name)


if __name__ == '__main__':
    main()
