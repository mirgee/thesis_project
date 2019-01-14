import logging
import math
import os
from functools import wraps

import click
import numpy as np
import pandas as pd

import mne
from lib.nolds.nolds import measures
from config import CHANNEL_NAMES, LABELED_ROOT, PROCESSED_ROOT, RAW_ROOT
from lib.HiguchiFractalDimension import hfd
from lib.nolitsa.nolitsa.utils import statcheck, gprange
from lib.nolitsa.nolitsa.d2 import c2_embed, d2, ttmle
from lib.nolitsa.nolitsa.delay import acorr, adfd, dmi
from lib.nolitsa.nolitsa.dimension import afn, fnn
from lib.nolitsa.nolitsa.lyapunov import mle_embed
from lib.nolitsa.nolitsa.surrogates import iaaft

registered_algos = []
measure_names = []


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
        def wrapper(data, *args, **kwargs):
            return f
    return decorator


# @register('fnn')
@log_result
def compute_dim_via_fnn(data, tau=3, window=50):
    R = 2.5
    A = 2.0
    dims_to_try = np.arange(1, 18)
    fnns = fnn(data, dim=dims_to_try, R=R, A=A, tau=tau, window=window,
               metric='euclidean')
    # Return the dimension with lowest number of FNN with both tests applied
    return np.argmin(fnns[2])


# @register('afn')
@log_result
def compute_dim_via_afn(data, tau=3, window=50):
    dims_to_try = np.arange(1, 18)
    E, _ = afn(
        data, tau=tau, dim=dims_to_try, window=window, metric='euclidean')
    E1 = E[1:] / E[:-1]
    return np.argmax(np.diff(E1) < 0.008)


# @register('tau_mi')
@log_result
def compute_tau_via_mi(data):
    def localmin(x):
        return (np.diff(np.sign(np.diff(x))) > 0).nonzero()[0] + 1
    maxtau = 20
    i = dmi(data, maxtau=maxtau)
    mi_mins = localmin(i)
    return mi_mins[0] if len(mi_mins) > 0 else np.nan


# @register('tau_acorr')
@log_result
def compute_tau_via_acorr(data):
    maxtau = 20
    r = acorr(data, maxtau=maxtau)
    return np.argmax(r < (1 - 1.0 / np.e)) + 1


# @register('tau_adfd')
@log_result
def compute_tau_via_adfd(data, dim=10):
    maxtau = 20
    disp = adfd(data, dim=dim, maxtau=maxtau)
    ddisp = np.diff(disp)
    forty = np.argmax(ddisp < 0.4 * ddisp[1])
    return ddisp[forty]


# @register('lyap')
@log_result
def compute_lyapunov(data, lib='nolitsa', autoselect_params=False):
    dim = 10
    tau = 7
    window = 50
    maxt = 15
    sampl_period = 1/250

    tau_dim_to_maxt = {
        (2, 5): 12,
        (2, 6): 14,
        (2, 7): 16,
        (2, 8): 18,
        (2, 9): 20,
        (2, 10): 22,
        (2, 11): 23,
        (2, 12): 25,
        (2, 13): 27,
        (2, 14): 28,
        (2, 15): 30,
        (2, 16): 33,
        (2, 17): 35,
        (2, 18): 37,
        (3, 5): 15,
        (3, 6): 18,
        (3, 7): 22,
        (3, 8): 25,
        (3, 9): 28,
        (3, 10): 30,
        (3, 11): 33,
        (3, 12): 38,
        (3, 13): 42,
        (3, 14): 46,
        (3, 15): 50,
        (3, 16): 55,
        (3, 17): 60,
        (3, 18): 65,
        (4, 5): 20,
        (4, 6): 24,
        (4, 7): 28,
        (4, 8): 32,
        (4, 9): 36,
        (4, 10): 40,
        (4, 11): 44,
        (4, 12): 48,
        (4, 13): 52,
        (4, 14): 56,
        (4, 15): 60,
        (4, 16): 64,
        (4, 17): 68,
        (4, 18): 72,
        (5, 5): 25,
        (5, 6): 30,
        (5, 7): 35,
        (5, 8): 40,
        (5, 9): 45,
        (5, 10): 50,
        (5, 11): 55,
        (5, 12): 60,
        (5, 13): 65,
        (5, 14): 70,
        (5, 15): 75,
        (5, 16): 80,
        (5, 17): 85,
        (5, 18): 90,
    }

    if autoselect_params:
        try:
            data, _ = find_least_stationary_window(
                data, win_width=15000, slide_width=100)
        except AssertionError:
            return np.nan

        tau_acorr = compute_tau_via_acorr(data)

        # tau_adfd = compute_tau_via_adfd(data)
        tau = max(min(tau_acorr, 5), 3)

        dim_fnn = compute_dim_via_fnn(data, tau, window)
        dim_afn = compute_dim_via_afn(data, tau, window)
        dim = max(min(math.ceil((dim_fnn + dim_afn) / 2), 18), 5)
        # Just to verify that adfd gives similar result for chosen dim
        tau_adfd = compute_tau_via_adfd(data, dim)

        maxt = tau_dim_to_maxt[(tau, dim)]
    else:
        maxt = dim * tau
    if lib == 'nolitsa':
        res = mle_embed(data, dim=[dim], tau=tau, window=window, maxt=maxt)[0]
        poly = np.polyfit(np.arange(len(res)), res/sampl_period, 1)
        lyap = poly[0]
    elif lib == 'nolds':
        lyap = measures.lyap_r(data, emb_dim=dim, lag=tau, min_tsep=window,
                            trajectory_len=maxt) / sampl_period
    else:
        raise NotImplementedError(f'Library {lib} not supported.')
    return lyap


# @register('corr')
@log_result
def compute_corr_dim(data, lib='nolds', autoselect_params=False):
    def smooth(y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    dim = 10
    dims = np.arange(2, 30 + 1)
    tau = 3
    window = 50
    maxt = 30
    rs = gprange(0.05, 10, 100)
    if lib == 'nolitsa':
        if autoselect_params:
            if len(data) > 15000:
                data, _ = find_least_stationary_window(
                    data, win_width=15000, slide_width=100)
            tau = compute_tau_via_acorr(data)
            # By passing integer r, we are using method for automatic selection
            # of range suggested by Galka (with our modified version of the
            # nolitsa library)
            rcs = c2_embed(data, dim=dims, tau=tau, r=100,
                           metric='chebyshev', window=window)
            d2s = [np.polyfit(np.log(r), np.log(c), 1)[0] for (r, c) in rcs
                   if len(r) > 0 and len(c) > 0]
            # d2s = smooth(d2s, 2)
            # diffs = np.diff(d2s)
            # sums = np.asarray(
            #     [np.abs(max(diffs[i-2:i+3]) - min(diffs[i-2:i+3]))
            #      for i in range(2, len(diffs)-3)])
            # return d2s[np.argmin(sums)]
            return d2s[np.argmax(d2s)]
        else:
            r, c = c2_embed(data, dim=[dim], tau=tau, r=rs,
                            metric='chebyshev', window=window)[0]
            if len(r) > 0 and len(c) > 0:
                return np.polyfit(np.log(r), np.log(c), 1)[0]
            else:
                return np.nan
    elif lib == 'nolds':
        prev_val = 0
        for dim in dims:
            corr_dim = measures.corr_dim(data, emb_dim=dim)
            if prev_val > corr_dim:
                corr_dim = prev_val
                break
            prev_val = corr_dim
        assert corr_dim != 0
    else:
        raise NotImplementedError(f'Library {lib} not supported.')
    return corr_dim


@register('dfa')
@log_result
def compute_dfa(data):
    from scipy.signal import butter, sosfilt, sosfreqz, hilbert

    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False,
                                     btype='band', output='sos')
        return sos

    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
        return y

    # lowcut = 3
    # highcut = 7
    # fs = 250

    # y = butter_bandpass_filter(data, lowcut, highcut, fs, order=4)
    # amplitude_envelope = np.abs(hilbert(y))

    nvals = measures.logarithmic_n(4, 320, 1.1)
    return measures.dfa(data, nvals=nvals, overlap=True,
                        offset_n=50)

# @register('hurst')
@log_result
def compute_hurst(data):
    return measures.hurst_rs(data)


# @register('sampen')
@log_result
def compute_sampen(data):
    return measures.sampen(data, emb_dim=2)


# @register('higu')
@log_result
def compute_higuchi(data):
    num_k = 50
    k_max = 50
    win_width = 5*250
    win_shift = 125
    win_beg = 0
    results = []
    while win_beg+win_width < len(data):
        results.append(
            hfd(data[win_beg:win_beg+win_width], num_k=num_k, k_max=k_max))
        win_beg += win_shift
    return sum(results)/len(results)


# @register('sigma_lyap')
@log_result
def compute_sigma_lyap(data):
    surrogates = [iaaft(data)[0] for _ in range(19)]
    true_lyap = compute_lyap(data)
    lyaps = [compute_lyap(surrogate) for surrogate in surrogates]
    return np.abs(np.mean(lyaps)-true_lyap) / np.std(lyaps)


# # @register('sigma_mle')
@log_result
def compute_sigma_mle(data):
    mle = np.empty(19)

    tau = 1
    window = 50
    dim = compute_embedding_dimension(data, tau, window)

    for k in range(19):
        y = iaaft(data)[0]
        r, c = c2_embed(y, dim=[dim], tau=tau, window=window)[0]

        r_mle, mle_surr = ttmle(r, c, zero=False)
        i = np.argmax(r_mle > 0.5 * np.std(y))
        mle[k] = mle_surr[i]

    r, c = c2_embed(data, dim=[dim], tau=tau, window=window)[0]

    r_mle, true_mle = ttmle(r, c, zero=False)
    i = np.argmax(r_mle > 0.5 * np.std(data))
    true_mle = true_mle[i]

    return np.abs(np.mean(mle)-true_mle) / np.std(mle)


@log_result
def find_least_stationary_window(x, win_width=5000, slide_width=100):
    assert len(x) > win_width
    start, end, min_start, min_end = 0, win_width, 0, 0
    min_p_value = np.inf

    while end < len(x):
        w = x[start:end]
        p_value = statcheck(w)[1]
        if p_value < min_p_value:
            min_p_value = p_value
            min_start, min_end = start, end
        start += slide_width
        end += slide_width

    return x[min_start:min_end], min_p_value
