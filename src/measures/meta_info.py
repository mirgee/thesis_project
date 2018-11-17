import logging
import os

import pandas as pd
import click

from config import LABELED_ROOT, PROCESSED_ROOT, RAW_ROOT
from data.utils import (get_meta_df, get_trial_index, 
                        get_trials, compute_label_thresholds)


def compute_three_class_label(file_path):
    trial, index, _ = get_trial_index(file_path)
    meta_df = get_meta_df()
    col = 'M_1' if trial == 'a' else 'M_4'
    label = meta_df.loc[index][col]

    # Precomputed label thresholds
    L, M = compute_label_thresholds()

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


def create_labels(input_path=PROCESSED_ROOT, output_path=LABELED_ROOT):
    logging.info('Creating labels...')
    idxs = pd.MultiIndex.from_product([list(range(1, 134)), ['a', 'b']],
                                      names=['patient', 'trial'])
    labels_df = pd.DataFrame(columns=['label_d', 'label_r', 'label_ba'], index=idxs)
    for file_name in os.listdir(input_path):
        if not file_name.endswith('.fif'):
            logging.info('Skipping file %s' % file_name)
            continue
        file_path = os.path.join(input_path, file_name)
        trial, index, _ = get_trial_index(file_name)
        labels_df.loc[(index, trial)]['label_d'] = compute_three_class_label(file_path)
        labels_df.loc[(index, trial)]['label_r'] = compute_binary_label(file_path)
        labels_df.loc[(index, trial)]['label_ba'] = 0 if trial == 'a' else 1
    logging.info('The resulting data: \n{}'.format(
        labels_df.describe()))
    logging.info('Saving label data as pickle...')
    labels_df.to_pickle(os.path.join(output_path, 'labels.pickle'))


@click.command()
@click.option('--out', type=str, default='training_with_meta.pkl')
def main():
    logging.basicConfig(level=logging.DEBUG)

    create_labels()


if __name__ == '__main__':
    main()
