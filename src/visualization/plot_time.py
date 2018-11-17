import logging
import os
import random as rd

import click
import mne
import pandas as pd
import numpy as np

from data.preprocess import preprocess_raw_mne_file
from data.data_files import CHANNEL_NAMES, DataKinds, files_builder


def interactive_plot_time(file):
    """Create an interactive figure visualizing all channels from a file."""

    def plot_mne_file(mne_file):
        fig = mne_file.plot(block=True, scalings='auto')
        fig.set_size_inches(18.5, 10.5, forward=True)

    sfreq = float(get_meta_df.loc[file.id, 'freq'])
    logging.info(f'Plotting file {file.name} of kind={file.kind}, sfreq={sfreq}...')
    info = mne.create_info(ch_names=CHANNEL_NAMES, sfreq=sfreq, ch_types='eeg')
    data = mne.io.RawArray(np.transpose(file.df.values), info)
    plot_mne_file(data)

    if kind == DataKinds.RAW:
        logging.info(f'Plotting processed file {file.name}...')
        processed = preprocess_raw_mne_file(data)
        plot_mne_file(processed)


@click.command()
@click.option('--kind', type=str, default=DataKinds.PROCESSED)
@click.option('--file', type=str, default='')
def main(kind, file):
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    if file:
        _, ext = os.path.splitext(file)
        file = files_builder(ext=ext).single_file(file)
        interactive_plot(file)
        return

    logger.info(f'Plotting EEG singals of kind {kind}...')

    for file in files_builder(kind):
        interactive_plot(file)


if __name__ == '__main__':
    main()
