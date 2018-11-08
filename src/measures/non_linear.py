import logging
import os

import pandas as pd

import mne
from config import CHANNEL_NAMES, LABELED_ROOT, PROCESSED_ROOT
from data.utils import df_from_fif, get_trial_index
from measures.utils import (FEATURE_NAMES, compute_corr_dim, compute_dfa,
                            compute_hurst, compute_lyapunov, compute_sampen)


def compute_nl(file_path):
    """Compute dict of non-linear features for trial recorded in file_name"""
    df = df_from_fif(file_path)
    new_row = {}

    for channel in CHANNEL_NAMES:
        values = df[channel].values
        new_row.update({
            (channel, 'lyap'): compute_lyapunov(values),
            (channel, 'corr'): compute_corr_dim(values),
            (channel, 'dfa'): compute_dfa(values),
            (channel, 'hurst'): compute_hurst(values),
            (channel, 'sampen'): compute_sampen(values),
        })

    return new_row


def create_training_data(input_path=PROCESSED_ROOT, output_path=LABELED_ROOT):
    """Create a dataframe with features and labels suitable for training."""
    logging.info('Creating training data...')

    cols = pd.MultiIndex.from_product([CHANNEL_NAMES, FEATURE_NAMES],
                                      names=['channel', 'feature'])
    idxs = pd.MultiIndex.from_product([list(range(1, 134)), ['a', 'b']],
                                      names=['patient', 'trial'])
    main_df = pd.DataFrame(columns=cols, index=idxs)

    for file_name in os.listdir(input_path):
        if not file_name.endswith('.fif'):
            logging.info('Skipping file %s' % file_name)
            continue
        file_path = os.path.join(input_path, file_name)
        new_row = compute_nl(file_path)
        trial, index, _ = get_trial_index(file_name)
        main_df.loc[(index, trial)] = pd.Series(new_row)
        logging.debug("Training dataframe after adding a row: \n%s" % main_df)

    logging.info('Saving training data as pickle...')
    pickle_name = 'training.pickle'
    main_df.to_pickle(os.path.join(output_path, pickle_name))


def main():
    logging.basicConfig(level=logging.DEBUG)
    mne.set_log_level(logging.ERROR)

    create_training_data()


if __name__ == '__main__':
    main()
