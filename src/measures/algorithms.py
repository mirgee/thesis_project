import logging
import os
from functools import wraps

import click
import numpy as np
import pandas as pd

import mne
import nolds
from config import CHANNEL_NAMES, LABELED_ROOT, PROCESSED_ROOT, RAW_ROOT
from lib.HiguchiFractalDimension import hfd
from lib.nolitsa.nolitsa.d2 import c2_embed, d2
from lib.nolitsa.nolitsa.dimension import fnn
from lib.nolitsa.nolitsa.lyapunov import mle_embed
from lib.nolitsa.nolitsa.delay import acorr, dmi

registered_algos = []
measure_names = []
EMBED_DIM = 10


def log_result(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        res = f(*args, **kwargs)
        logging.debug(f'{f.__name__} returned: {res}')
        return res
    return wrapper


def register(algo_name):
    assert algo_name, 'Algorithm name is mandatory for registration.'
    measure_names.append(algo_name)

    def decorator(f):
        f.algo_name = algo_name
        registered_algos.append(f)

        @wraps(f)
        def wrapper(*args, **kwargs):
            return f
    return decorator


# @register('emdim')
@log_result
def compute_embedding_dimension(data, tau, window):
    R = 3.0
    A = 0.5
    dims_to_try = np.arange(1, 15)
    fnns = fnn(data, dim=dims_to_try, R=R, A=A, tau=tau, window=window,
               metric='euclidean')
    # Return the dimension with lowest number of FNN with both tests applied
    return np.argmin(fnns[2])


@register('tau_mi')
@log_result
def compute_tau_via_mi(data):
    def localmin(x):
        return (np.diff(np.sign(np.diff(x))) > 0).nonzero()[0] + 1
    maxtau = 20
    i = dmi(data, maxtau=maxtau)
    mi_mins = localmin(i)
    return mi_mins[0] if len(mi_mins) > 0 else np.nan


@register('tau_acorr')
@log_result
def compute_tau_via_acorr(data):
    maxtau = 20
    r = acorr(data, maxtau=maxtau)
    return np.argmax(r < (1 - 1.0 / np.e))


# @register('lyap')
@log_result
def compute_lyapunov(data, lib='nolitsa', use_fnn=True):
    dim = 8
    tau = 1
    window = 50
    maxt = 15
    sampl_period = 1/250
    if use_fnn:
        # When no lag specified, it is found via autocorrelation method
        dim = compute_embedding_dimension(data, tau, window)
    if lib == 'nolitsa':
        res = mle_embed(data, dim=[dim], tau=tau, window=window, maxt=maxt)[0]
        poly = np.polyfit(np.arange(len(res)), res, 1)
        lyap = poly[0]
    elif lib == 'nolds':
        lyap = nolds.lyap_r(data, emb_dim=dim, lag=tau, min_tsep=window,
                            trajectory_len=maxt) / sampl_period
    else:
        raise NotImplementedError(f'Library {lib} not supported.')
    return lyap


# @register('corr')
@log_result
def compute_corr_dim(data, lib='nolitsa'):
    dims = list(range(3, 10))
    tau = 3
    window = 100
    maxt = 30
    if lib == 'nolitsa':
        for dim in dims:
            r, c = c2_embed(data, dim=[dim], tau=tau, r=100,
                            metric='chebyshev', window=window)[0]
            d = d2(r, c, hwin=3)
            return sum(d[:maxt])/maxt
    elif lib == 'nolds':
        prev_val = 0
        for dim in dims:
            corr_dim = nolds.corr_dim(data, emb_dim=dim)
            if prev_val > corr_dim:
                corr_dim = prev_val
                break
            prev_val = corr_dim
        assert corr_dim != 0
    else:
        raise NotImplementedError(f'Library {lib} not supported.')
    return corr_dim


# @register('dfa')
@log_result
def compute_dfa(data):
    return nolds.dfa(data)


# @register('hurst')
@log_result
def compute_hurst(data):
    return nolds.hurst_rs(data)


# @register('sampen')
@log_result
def compute_sampen(data):
    return nolds.sampen(data, emb_dim=EMBED_DIM)


# @register('higu')
@log_result
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
