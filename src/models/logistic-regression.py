import numpy as np
import pandas as pd
import click
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets, metrics
from sklearn.metrics import confusion_matrix, classification_report,
    roc_auc_score, roc_curve
import statsmodels.api as sm
import os
import logging

from config import LABELED_ROOT, VISUAL_ROOT

TEST_SIZE = 0.3


def train(X_train, y_train):
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    return logreg


def summary(X, y):
    logit_model = sm.Logit(y, X)
    result = logit_model.fit()
    summary = result.summary2()
    logging(f'Summary for logistic regression classifier: \n{summary}')


def evaluate(logreg, X_test, y_test, y_pred):
    logging.info('Evaluating logistic regression classifier...')

    y_pred = logreg.predict(X_test)
    logging.info('Accuracy on the test set: {:.2f}'.format(logreg.score(
        X_test, y_test)))

    confusion_matrix = confusion_matrix(y_test, y_pred)
    logging.info(f'Confusion matrix: \n{confusion_matrix}')

    report = classification_report(y_test, y_pred)
    logging.info(f'Classification report: \n{report}')


def plot_roc_curve(X_test, y_test):
    logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(VISUAL_ROOT, 'Log_ROC'))

    plt.show()


def logistic_regression():
    logging.info('Performing logistic regression on the dataset...')

    file_name = 'training.pickle'
    df = pd.read_pickle(os.path.join(LABELED_ROOT, file_name))

    X = df.values()[:-1]
    y = df.values()[-1]

    summary(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=TEST_SIZE,
                                                        random_state=0)
    logreg = train(X_train, y_train)

    evaluate(log_reg, X_test, y_test, y_pred)

    plot_roc_curve(X_test, y_test)


@click.command()
def main(input_path=LABELED_ROOT):
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    logistic_regression()

if __name__ == '__main__':
    main()
