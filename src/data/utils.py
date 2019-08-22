import os

import numpy as np
import pandas as pd

import mne
from config import DATA_ROOT, CHANNEL_NAMES, META_COLUMN_NAMES, LABELED_ROOT


def df_from_tdt(file_path):
    """Create a dataframe from a .tdt (raw) file."""
    return pd.read_csv(
        file_path, sep='\t', names=CHANNEL_NAMES, skiprows=[0])


def df_from_fif(file_path):
    """Create a dataframe from a .fif (preprocessed) file."""
    raw_fif = mne.io.read_raw_fif(file_path)
    t = pd.DataFrame(raw_fif.get_data())
    return pd.DataFrame(np.transpose(t.values), columns=CHANNEL_NAMES)


def data_from_npy(file_path):
    """Load data from file_path."""
    return np.load(file_path)


def get_meta_df():
    """Create a dataframe from the .xlsx file describing the individual
    recordings."""
    raw_root = os.path.abspath(os.path.join(DATA_ROOT, 'raw'))
    meta_file_name = 'DEP-POOL_Final_144.xlsx'
    return pd.read_excel(
        os.path.join(raw_root, meta_file_name), names=META_COLUMN_NAMES)


def mne_from_file(file):
    """Get a MNE instance based on supplied File file."""
    sfreq = float(get_meta_df().loc[file.id, 'freq'])
    info = mne.create_info(ch_names=CHANNEL_NAMES, sfreq=sfreq, ch_types='eeg')
    return mne.io.RawArray(np.transpose(file.df.values), info)


def get_recording_length(file_path, df, seconds):
    duration = get_duration(file_path, df)
    if duration < seconds:
        raise IndexError
    sfreq = get_sampling_frequency(file_path)
    return seconds*sfreq


def get_duration(file_path, df):
    num_rows = len(df.index)
    sfreq = get_sampling_frequency(file_path)
    return (num_rows-1)/sfreq


def get_indexes_and_trials(input_path):
    its = []
    for file_name in os.listdir(input_path):
        if not file_name.endswith('.fif') and not file_name.endswith('.tdt'):
            continue
        trial = get_trial(file_name)
        index = get_index(file_name)
        its.append((index, trial))
    return its


def get_index(file_name):
    """Get index, i.e. the patient number, of the recording under supplied
    path."""
    return int(_get_no_ext_file_name(file_name)[:-1])


def get_trial(file_name):
    """Get trial of the recording under supplied path, i.e. is it a before /
    after treatment recording."""
    return _get_no_ext_file_name(file_name)[-1]


def _get_no_ext_file_name(file_name):
    no_ext_file_name = os.path.split(os.path.splitext(file_name)[0])[1]
    i = no_ext_file_name.find('-')
    if i > 0:
        no_ext_file_name = no_ext_file_name[:i]
    return no_ext_file_name


def get_sampling_frequency(file_path):
    meta_df = get_meta_df()
    index = get_index(file_path)
    return meta_df.iloc[index-1]['freq']


def raw_mne_from_tdt(file_path):
    assert(file_path.endswith('.tdt'))
    sampling_freq = get_sampling_frequency(file_path)

    df = df_from_tdt(file_path)
    info = mne.create_info(ch_names=CHANNEL_NAMES, sfreq=sampling_freq,
                           ch_types='eeg')
    data = mne.io.RawArray(np.transpose(df.values), info)
    return data


def prepare_dfs(col='lyap', kind='processed'):
    bands = {'delta', 'theta', 'alpha', 'beta', 'gamma'}
    band = None
    if col in bands:
        band = col
        col = 'ampl'
    df = pd.read_pickle(os.path.join(
        LABELED_ROOT, kind, col, f'training_{col}.pkl'))
    df_bef = pd.read_pickle(
        os.path.join(LABELED_ROOT, kind, col, f'{col}_bef.pkl'))
    df_aft = pd.read_pickle(
        os.path.join(LABELED_ROOT, kind, col, f'{col}_aft.pkl'))
    if band is not None:
        df = df.loc[:, (slice(None), band)]
        df_bef = df_bef.loc[:, (slice(None), band)]
        df_aft = df_aft.loc[:, (slice(None), band)]
    return df, df_bef, df_aft


def get_metapkl():
    return pd.read_pickle(os.path.join(LABELED_ROOT, 'processed', 'meta',
                                       'meta.pkl'))


def prepare_resp_non(col='lyap'):
    df, df_bef, df_aft = prepare_dfs(col)
    return df[df['resp'] == 1], df[df['resp'] == 0]


def prepare_dep_non(col='lyap'):
    df, df_bef, df_aft = prepare_dfs(col)
    return df[df['dep'] == 1], df[df['dep'] == -1]
