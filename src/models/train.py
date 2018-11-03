import logging
import os

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, metrics, svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

import statsmodels.api as sm
from config import LABELED_ROOT, VISUAL_ROOT

TEST_SIZE = 0.3


def train_lr(X_train, y_train):
    # C is inverse of regularization strength
    logreg = LogisticRegression(C=1e5, solver='lbgfs', multi_class='multinomial')
    return logreg.fit(X_train, y_train)


def train_svm(X_train, y_train):
    svm = svm.SVC(kernel='rbf', gamma='scale')
    return svm.fit(X_train, y_train)


def summary(X, y):
    logit_model = sm.Logit(y, X)
    result = logit_model.fit()
    summary = result.summary2()
    logging.info(f'Summary for logistic regression classifier: \n{summary}')


def evaluate(model, X_test, y_test, y_pred):
    logging.info('Evaluating classifier...')

    y_pred = model.predict(X_test)
    logging.info('Accuracy on the test set: {:.2f}'.format(model.score(
        X_test, y_test)))

    confusion_matrix = confusion_matrix(y_test, y_pred)
    logging.info(f'Confusion matrix: \n{confusion_matrix}')

    report = classification_report(y_test, y_pred)
    logging.info(f'Classification report: \n{report}')


def plot_roc_curve(X_test, y_test):
    logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
    fpr, tpr, thresholds = roc_curve(
        y_test, logreg.predict_proba(X_test)[:, 1])

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


def train(classifier):
    logging.info('Performing logistic regression on the dataset...')

    df_X = pd.read_pickle(os.path.join(LABELED_ROOT, 'splits.pickle'))
    # Feature selection
    rows = [row for row in df_X.index if row[0].startswith('b')]
    # Only total param
    df_X.loc[rows, (slice(None), [4], slice(None))].unstack()

    df_X = df_X[['lyap', 'corr', 'dfa', 'hurst']][4]
    df_y = pd.read_pickle(os.path.join(LABELED_ROOT, 'labels.pickle'))
    df = df_X.join(df_y)

    logging.info(f'The dataframe used for training: \n{df}')

    X = df.values[:-1]
    y = df.values[-1]

    # summary(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=TEST_SIZE,
                                                        random_state=0)
    if classifier == 'logistic-regression':
        model = train_lr(X_train, y_train)
    elif classifier == 'svm':
        model = train_svm(X_train, y_train)
    else:
        raise NotImplemented(f'Classifier {classifier} not implemented.')

    evaluate(model, X_test, y_test, y_pred)

    plot_roc_curve(X_test, y_test)


@click.command()
@click.argument('classifier', default='logistic-regression')
def main(classifier, input_path=LABELED_ROOT):
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    train(classifier)


if __name__ == '__main__':
    main()
