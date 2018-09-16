import pandas as pd
import mne

from ..config import CHANNEL_NAMES, DATA_ROOT

def read_raw(file_name):
    df = pd.read_table(file_name, sep='\t', names=CHANNEL_NAMES, skiprows=[0])
    return df

def read_mne_raw_data(file_name, sampling_frequency):
    df = read_raw(file_name)
    info = mne.create_info(ch_names=CHANNEL_NAMES, sfreq=sampling_frequency,
                           ch_types = 'eeg')
    data = mne.io.RawArray(np.transpose(df.values), info)
    return data
