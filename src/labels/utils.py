import logging
import os

import click
import numpy as np
import pandas as pd

import mne
import nolds
from config import CHANNEL_NAMES, LABELED_ROOT, PROCESSED_ROOT, RAW_ROOT
from data.utils import df_from_fif, get_meta_df, get_trial_index, get_trials
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
    new_row = {}

    for channel in CHANNEL_NAMES:
        values = df[channel].values
        new_row.update({
            (channel, 'lyap'): compute_lyapunov(values),
            (channel, 'corr'): compute_corr_dim(values),
            (channel, 'dfa'): compute_dfa(values),
            (channel, 'hurst'): compute_hurst(values),
        })

    return new_row


def create_training_data(input_path=PROCESSED_ROOT, output_path=LABELED_ROOT):
    """Create a dataframe with features and labels suitable for training."""
    logging.info('Creating training data...')

    cols = pd.MultiIndex.from_product([CHANNEL_NAMES, FEATURE_NAMES],
                                      names=['channel', 'feature'])
    trials = get_trials(RAW_ROOT)
    main_df = pd.DataFrame(columns=cols, index=trials)

    for file_name in os.listdir(input_path):
        if not file_name.endswith('.fif'):
            logging.info('Skipping file %s' % file_name)
            continue
        file_path = os.path.join(input_path, file_name)
        new_row = compute_nl(file_path)
        _, _, trial = get_trial_index(file_name)
        main_df.loc[trial] = pd.Series(new_row)
        logging.debug("Training dataframe after adding a row: \n%s" % main_df)

    logging.info('Saving training data as pickle...')
    pickle_name = 'training.pickle'
    main_df.to_pickle(os.path.join(output_path, pickle_name))
