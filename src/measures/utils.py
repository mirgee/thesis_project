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
from lib.HiguchiFractalDimension import hfd

FEATURE_NAMES = ['lyap', 'corr', 'dfa', 'hurst', 'sampen']
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


def compute_sampen(data):
    sampen = nolds.sampen(data, emb_dim=EMBED_DIM)
    logging.debug('Computed sampen: %f' % sampen)
    return sampen


def compute_higuchi(data):
    num_k = 50
    k_max = 50
    win_width = 1500
    win_shift = 200
    win_beg = 0
    results = []
    while win_beg+win_width < len(data):
        results.append(
            hfd(data[win_beg:win_beg+win_width], num_k=num_k, k_max=k_max))
        win_beg += win_shift
    return sum(results)/len(results)


def compute_embedding_dimension(data):
    dims_to_try = np.arange(1, 15)
    fnns = fnn(data, dim=dims_to_try)
    # Return the dimension with lowest number of FNN with both tests applied
    m = np.argmin(fnns[2])
    logging.debug(f'Choosing dimension {m}')
    return m
