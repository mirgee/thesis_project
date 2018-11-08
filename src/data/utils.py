import logging
import os

import numpy as np
import pandas as pd

import mne
from config import CHANNEL_NAMES, META_COLUMN_NAMES, META_FILE_NAME, RAW_ROOT


def df_from_tdt(file_path, seconds=None):
    df = pd.read_table(file_path, sep='\t', names=CHANNEL_NAMES, skiprows=[0])
    n = len(df) if seconds is None else get_recording_length(file_path, df,
                                                             seconds)
    return df.iloc[:n]


def df_from_fif(file_path, seconds=None):
    raw_fif = mne.io.read_raw_fif(file_path)
    t = pd.DataFrame(raw_fif.get_data())
    df = pd.DataFrame(np.transpose(t.values), columns=CHANNEL_NAMES)
    sfreq = int(raw_fif.info["sfreq"])
    dur = int(raw_fif.info["file_id"]['secs'])
    n = len(df) if seconds is None else sfreq*seconds
    logging.debug(f'{file_path}: {sfreq} Hz, {dur} s')
    return df.iloc[:n]


def get_meta_df():
    return pd.read_excel(os.path.join(RAW_ROOT, META_FILE_NAME), index_col='ID',
                         names=META_COLUMN_NAMES)


def get_recording_length(file_path, df, seconds):
    duration = get_duration(file_path, df)
    if duration < seconds:
        raise IndexError
    sfreq = get_sampling_frequency(file_path)
    return seconds*sfreq


def get_duration(file_path, df):
    num_rows = len(df.index)
    sfreq = get_sampling_frequency(file_path)
    _, _, trial = get_trial_index(file_path)
    return (num_rows-1)/sfreq


def compute_label_thresholds():
    meta_df = get_meta_df()
    s = sorted(meta_df['M_1'].append(meta_df['M_4']).values)
    n = len(s)
    L = s[n // 3]
    M = s[2*n // 3]

    return L, M


def remove_extension(file_path):
    return os.path.splitext(file_path.split(os.sep)[-1])[0]


def get_trial_index(file_path):
    no_ext_file_name = remove_extension(file_path)

    trial = no_ext_file_name[-1]
    patient_id = no_ext_file_name[:-1]

    return trial, int(patient_id), str(no_ext_file_name)


def get_trials(input_path):
    trials = []
    for file_name in os.listdir(input_path):
        if not file_name.endswith('.fif') and not file_name.endswith('.tdt'):
            continue
        _, _, trial_index = get_trial_index(file_name)
        trials.append(trial_index)
    assert trials
    return trials


def get_sampling_frequency(file_path):
    meta_df = get_meta_df()
    trial_num, index, _ = get_trial_index(file_path)
    return meta_df.iloc[index-1]['freq']


def raw_mne_from_tdt(file_path):
    assert(file_path.endswith('.tdt'))
    sampling_freq = get_sampling_frequency(file_path)

    df = df_from_tdt(file_path)
    info = mne.create_info(ch_names=CHANNEL_NAMES, sfreq=sampling_freq,
                           ch_types='eeg')
    data = mne.io.RawArray(np.transpose(df.values), info)
    return data
