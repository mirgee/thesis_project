import logging
import os

import pandas as pd

from config import LABELED_ROOT, PROCESSED_ROOT, RAW_ROOT
from data.utils import get_meta_df, get_trial_index, get_trials


def compute_three_class_label(file_path):
    trial, index, _ = get_trial_index(file_path)
    meta_df = get_meta_df()
    col = 'M_1' if trial == 'a' else 'M_4'
    label = meta_df.loc[index][col]

    # Precomputed label thresholds
    L, M = 38, 48

    if label <= L:
        return 'L'
    elif label <= M:
        return 'M'
    else:
        return 'H'


def compute_binary_label(file_path):
    trial, index, _ = get_trial_index(file_path)
    meta_df = get_meta_df()
    return meta_df.loc[index]['RESP_4W']


def create_labels(input_path=PROCESSED_ROOT, output_path=LABELED_ROOT,
                  three_class=False):
    logging.info('Creating labels...')
    trials = get_trials(RAW_ROOT)
    labels_df = pd.DataFrame(columns=['label'], index=trials)
    for file_name in os.listdir(input_path):
        if not file_name.endswith('.fif'):
            logging.info('Skipping file %s' % file_name)
            continue
        file_path = os.path.join(input_path, file_name)
        _, _, trial = get_trial_index(file_name)
        if not three_class:
            labels_df.loc[trial]['label'] = compute_binary_label(file_path)
        else:
            labels_df.loc[trial]['label'] = \
                compute_three_class_label(file_path)
    logging.info('Saving label data as pickle...')
    pickle_name = 'labels.pickle'
    labels_df.to_pickle(os.path.join(output_path, pickle_name))


def main():
    logging.basicConfig(level=logging.DEBUG)

    create_labels()
