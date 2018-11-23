import logging
import os

import click
import pandas as pd

import measures.algorithms as algos
import mne
from config import LABELED_ROOT
from data.data_files import CHANNEL_NAMES, DataKind, files_builder


def compute_nl(df):
    """Compute dict of non-linear features for trial recorded in file_name"""
    new_row = {}

    for channel in CHANNEL_NAMES:
        values = df[channel].values
        new_row.update({
            (channel, f.algo_name): f(values) for f in algos.registered_algos
        })

    return new_row


def create_training_data(output_path, kind):
    """Create a dataframe with features and labels suitable for training."""
    logging.info('Creating training data.')

    cols = pd.MultiIndex.from_product([CHANNEL_NAMES, algos.measure_names],
                                      names=['channel', 'measure'])
    idxs = pd.MultiIndex.from_product([list(range(1, 134)), ['a', 'b']],
                                      names=['patient', 'trial'])
    main_df = pd.DataFrame(columns=cols, index=idxs)

    for file in files_builder(kind):
        new_row = compute_nl(file.df)
        main_df.loc[(file.id, file.trial)] = pd.Series(new_row)
        logging.debug("New row: \n%s" % new_row)

        logging.debug(f'Saving training data at {output_path}.')
        main_df.to_pickle(output_path)


@click.command()
@click.option('--fname', type=str, default='measures.pkl')
@click.option('--kind', type=str, default=DataKind.PROCESSED)
def main(fname, kind):
    logging.basicConfig(level=logging.DEBUG)
    mne.set_log_level(logging.ERROR)

    output_path = os.path.join(LABELED_ROOT, kind)
    assert os.path.exists(output_path), output_path
    fpath = os.path.join(output_path, fname)
    if os.path.isfile(fpath):
        logging.warning(f'File {fpath} exists and will be overwritten.')
    create_training_data(fpath, DataKind(kind))


if __name__ == '__main__':
    main()
