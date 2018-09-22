import logging
import os
import random as rd

import click
import mne
import pandas as pd
import numpy as np

from config import CHANNEL_NAMES, DATA_ROOT, PROCESSED_ROOT, RAW_ROOT
from data.utils import df_from_fif, df_from_tdt, get_sfreq

def interactive_plot(input_file, kind, apply_proj=False):
    """Create an interactive figure visualizing all channels from a file."""
    df = df_from_fif(input_file) if kind=='PROCESSED' else \
        df_from_tdt(input_file)

    sfreq = get_sfreq(input_file)
    info = mne.create_info(ch_names=CHANNEL_NAMES, sfreq=sfreq, ch_types='eeg')
    data = mne.io.RawArray(np.transpose(df.values), info)

    logging.info(f'Plotting file {input_file} of kind={kind}, sfreq={sfreq}')

    if apply_proj:
        data.set_eeg_reference('average', projection=True)
        data.apply_proj()

    fig = data.plot(block=True, scalings='auto')
    fig.set_size_inches(18.5, 10.5, forward=True)

    return fig


def examine_all(kind='RAW', proj=False):
    input_folder = PROCESSED_ROOT if kind=='PROCESSED' else RAW_ROOT

    to_examine = os.listdir(input_folder)
    rd.shuffle(to_examine)

    for file_name in to_examine:
        interactive_plot(os.path.join(input_folder, file_name), kind, proj)

@click.command()
@click.option('--kind', type=str, default='RAW')
@click.option('--proj', type=bool, default=False)
def main(kind, proj):
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    logger.info(f'Plotting EEG singals of kind {kind}...')

    examine_all(kind, proj)

if __name__ == '__main__':
    main()
