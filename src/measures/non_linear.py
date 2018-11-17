import logging
import os

import click
import pandas as pd

import measures.algorithms
import mne
from config import LABELED_ROOT
from data.data_files import CHANNEL_NAMES, DataKinds, files_builder
from measures.algorithms import measure_names, registered_algos


def compute_nl(df):
    """Compute dict of non-linear features for trial recorded in file_name"""
    new_row = {}

    for channel in CHANNEL_NAMES:
        values = df[channel].values
        new_row.update({
            (channel, f.algo_name): f(values) for f in registered_algos
        })

    return new_row


def create_training_data(output_path):
    """Create a dataframe with features and labels suitable for training."""
    logging.info('Creating training data.')

    cols = pd.MultiIndex.from_product([CHANNEL_NAMES, measure_names],
                                      names=['channel', 'measure'])
    idxs = pd.MultiIndex.from_product([list(range(1, 134)), ['a', 'b']],
                                      names=['patient', 'trial'])
    main_df = pd.DataFrame(columns=cols, index=idxs)

    for df_file, index, trial in files_builder(DataKinds.PROCESSED):
        new_row = compute_nl(df_file)
        main_df.loc[(index, trial)] = pd.Series(new_row)
        logging.debug("New row: \n%s" % new_row)

        logging.debug(f'Saving training data at {output_path}.')
        main_df.to_pickle(output_path)


@click.command()
@click.option('--out', type=str, default='measures.pkl')
def main(out):
    logging.basicConfig(level=logging.DEBUG)
    mne.set_log_level(logging.ERROR)

    output_path = os.path.join(LABELED_ROOT, out)
    create_training_data(output_path)


if __name__ == '__main__':
    main()
