import logging
import os
import random as rd

import click
import mne
import pandas as pd
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt

from config import CHANNEL_NAMES, DATA_ROOT, PROCESSED_ROOT, RAW_ROOT
from data.utils import df_from_fif, df_from_tdt, get_sampling_frequency
from data.preprocess import preprocess_raw_mne_file


def plot_psd(X, label, Fs, NFFT, color=None):

    noverlap = int(NFFT * 0.8)

    freqs, psd = welch(X, fs=Fs, window='hann', nperseg=NFFT,
                       noverlap=noverlap)

    # print(len(freqs))
    # print(len(psd))

    f = freqs[freqs > np.zeros(len(freqs))]
    psd = psd[freqs > np.zeros(len(freqs))]

    plt.plot(np.log10(f), 10 * np.log10(psd.ravel()), label=label,
             color=color)


def interactive_plot_freq(input_file, kind, apply_proj=False):
    """Create an interactive figure visualizing all channels from a file."""
    df = df_from_fif(input_file) if kind=='PROCESSED' else \
        df_from_tdt(input_file)

    if kind == 'RAW':
        sfreq = get_sampling_frequency(input_file)
    else:
        sfreq = 250
    info = mne.create_info(ch_names=CHANNEL_NAMES, sfreq=sfreq, ch_types='eeg')
    data = mne.io.RawArray(np.transpose(df.values), info)
    processed = preprocess_raw_mne_file(data, apply_proj)

    logging.info(f'Plotting frequency components of  {input_file} of'
                 'kind={kind}, sfreq={sfreq}...')

    # data.plot_psd()
    # processed.plot_psd()

    # add some plotting parameter
    # decim_fit = 100  # we lean a purely spatial model, we don't need all samples
    # decim_show = 10  # we can make plotting faster
    n_fft = 2 ** 13  # let's use long windows to see low frequencies


    for i, channel in enumerate(CHANNEL_NAMES):
        plt.figure(figsize=(9, 6))
        values = df[channel].values
        logging.info(f'Plotting file {input_file}...')
        plot_psd(values, Fs=sfreq, NFFT=n_fft, label='EEG', color='black')

        if kind == 'RAW':
           plot_psd(processed[i][0][0], Fs=sfreq, NFFT=n_fft,
                    label='EEG-processed', color='orange')

        plt.legend()
        plt.xticks(np.log10([0.1, 1, 10, 100]), [0.1, 1, 10, 100])
        plt.xlim(np.log10([0.1, 300]))
        plt.xlabel('log10(frequency) [Hz]')
        plt.ylabel('Power Spectral Density [dB]')
        plt.grid()
        plt.show()


def examine_all_freq(kind='RAW', proj=False):
    input_folder = PROCESSED_ROOT if kind=='PROCESSED' else RAW_ROOT

    to_examine = os.listdir(input_folder)
    rd.shuffle(to_examine)

    for file_name in to_examine:
        interactive_plot_freq(os.path.join(input_folder, file_name), kind, proj)


@click.command()
@click.option('--kind', type=str, default='RAW')
@click.option('--proj', type=bool, default=False)
def main(kind, proj):
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    logger.info(f'Plotting PSD of EEG singals of kind {kind}...')

    examine_all_freq(kind, proj)

if __name__ == '__main__':
    main()
