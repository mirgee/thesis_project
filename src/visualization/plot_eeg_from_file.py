import logging
import os

import click
import mne
import pandas as pd
import numpy as np

from .config import CHANNEL_NAMES, DATA_ROOT

def get_figure(input_file, names):
    """Create an matplotlib figure visualizing all channels from a file."""

    df = pd.read_table(input_file, sep='\t', names=names, skiprows=[0])

    info = mne.create_info(ch_names=names, sfreq=250, ch_types='eeg')
    data = mne.io.RawArray(np.transpose(df.values), info)

    data.set_eeg_reference('average', projection=True)
    data.apply_proj()

    fig = data.plot(block=True, lowpass=40, scalings='auto')
    fig.set_size_inches(18.5, 10.5, forward=True)

    return fig

@click.command()
@click.argument('input_file', type=click.Path(exists=False, dir_okay=False))
@click.option('--output_file', type=click.Path(writable=True, dir_okay=False),
              default=None)
@click.option('--show', default=True)
def main(input_file, output_file=None, show=False):
    logger = logging.getLogger(__name__)
    logger.info("Plotting EEG singals from {input_file} to {output_file}...")

    input_file_abs = os.path.abspath(os.path.join(DATA_ROOT, input_file))

    fig = get_figure(input_file_abs, CHANNEL_NAMES)
    if show:
        fig.plot()
    if output_file:
        fig.save(output_file)

if __name__ == '__main__':
    main()
