import pandas as pd
import numpy as np
import mne
import os

from config import (CHANNEL_NAMES, DATA_ROOT, RAW_ROOT, META_FILE_NAME,
META_COLUMN_NAMES)


def df_from_tdt(file_path):
    df = pd.read_table(file_path, sep='\t', names=CHANNEL_NAMES, skiprows=[0])
    return df


def df_from_fif(file_path):
    raw_fif = mne.io.read_raw_fif(file_path)
    t = pd.DataFrame(raw_fif.get_data())
    df = pd.DataFrame(np.transpose(t.values), columns=CHANNEL_NAMES)
    return df


def get_meta_df():
    return pd.read_excel(os.path.join(RAW_ROOT, META_FILE_NAME), index_col='ID',
                        names=META_COLUMN_NAMES)


def compute_label_thresholds():
    meta_df = get_meta_df()
    s = sorted(meta_df['M_1'].values + meta_df['M_4'].values)
    n = len(s)
    L = s[n // 3]
    M = s[2*n // 3]

    return L, M


def get_sfreq(file_name):
    trial, index = get_trial_index(file_name)
    meta_df = get_meta_df()

    return meta_df.loc[index]['freq']


def remove_extension(file_path):
    return os.path.splitext(file_path.split(os.sep)[-1])[0] 


def get_trial_index(file_path):
    no_ext_file_name = remove_extension(file_path)

    trial_num = no_ext_file_name[-1]
    index = no_ext_file_name[:-1]

    return trial_num, int(index), '-'.join((trial_num, index))


def get_trials(input_path):
    trials = []
    for file_name in os.listdir(input_path):
        if not file_name.endswith('.fif'):
            continue
        _, _, trial_index = get_trial_index(file_name)
        trials.append(trial_index)
    return trials


def raw_mne_from_tdt(file_path):
    assert(file_path.endswith('.tdt'))
    meta_df = get_meta_df()
    trial_num, index = get_trial_index(file_path)
    sampling_freq = meta_df.iloc[index-1]['freq']
    
    df = df_from_tdt(file_path)
    info = mne.create_info(ch_names=CHANNEL_NAMES, sfreq=sampling_freq,
                           ch_types = 'eeg')
    data = mne.io.RawArray(np.transpose(df.values), info)
    return data

