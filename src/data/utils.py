import os

import numpy as np
import pandas as pd

import mne
from config import CHANNEL_NAMES, META_COLUMN_NAMES, META_FILE_NAME, RAW_ROOT


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
