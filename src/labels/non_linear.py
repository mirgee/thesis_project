import nolds
import pandas as pd
import logging
import os
import numpy as np
import mne

from data.utils import df_from_fif, get_trial_index, get_meta_df, get_trials
from config import CHANNEL_NAMES, PROCESSED_ROOT, LABELED_ROOT, RAW_ROOT
from lib.nolitsa.dimension import fnn

FEATURE_NAMES = ['lyap', 'corr', 'dfa', 'hurst']
EMBED_DIM = 10


def compute_lyapunov(data, use_fnn=False):
    if use_fnn:
        # When no lag specified, it is found via autocorrelation method
        m = compute_embedding_dimension(data)
    lyap = nolds.lyap_r(data, emb_dim=EMBED_DIM)
    logging.debug('Computed LE: %f' % lyap)
    return lyap


def compute_corr_dim(data):
    corr_dim = nolds.corr_dim(data, emb_dim=EMBED_DIM)
    logging.debug('Computed CD: %f' % corr_dim)
    return corr_dim


def compute_dfa(data):
    dfa = nolds.dfa(data)
    logging.debug('Computed DFA: %f' % dfa)
    return dfa


def compute_hurst(data):
    hurst = nolds.hurst_rs(data)
    logging.debug('Computed hurst: %f' % hurst)
    return hurst


def compute_embedding_dimension(data):
    dims_to_try = np.arange(1, 15)
    fnns = fnn(data, dim=dims_to_try)
    # Return the dimension with lowest number of FNN with both tests applied
    m = np.argmin(fnns[2])
    logging.debug(f'Choosing dimension {m}')
    return m


def compute_nl(file_path):
    """Compute dict of non-linear features for trial recorded in file_name"""
    df = df_from_fif(file_path)
    features = []

    for channel in CHANNEL_NAMES:
        values = df[channel].values
        lyap = compute_lyapunov(values)
        corr = compute_corr_dim(values)
        dfa = compute_dfa(values)
        hurst = compute_hurst(values)

        features.extend([lyap, corr, dfa, hurst])

    return features


def compute_label(file_path):
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


def create_training_data(input_path=PROCESSED_ROOT, output_path=LABELED_ROOT):
    """Create a dataframe with features and labels suitable for training."""
    logging.info('Creating training data...')

    idx_name = 'trials'
    cols = ([idx_name] +
            ['-'.join((f, c)) for c in CHANNEL_NAMES for f in FEATURE_NAMES] +
            ['label'])
    trials = get_trials(RAW_ROOT)
    main_df = pd.DataFrame(columns=cols, index=trials)

    for file_name in os.listdir(input_path):
        if not file_name.endswith('.fif'):
            logging.info('Skipping file %s' % file_name)
            continue

        file_path = os.path.join(input_path, file_name)

        features = compute_nl(file_path)
        label = compute_label(file_path)

        logging.info("Computed vector of features %s" % features + [label])

        _, _, trial = get_trial_index(file_name)
        main_df.append(
            pd.Series(features + [label], name=trial))
        logging.debug("Training dataframe after adding a row: \n%s" % main_df)

    logging.info('Saving training data as pickle...')
    pickle_name = 'training.pickle'
    main_df.to_pickle(os.path.join(output_path, pickle_name))


def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)
    mne.set_log_level(logging.WARNING)

    create_training_data()

if __name__ == '__main__':
    main()
