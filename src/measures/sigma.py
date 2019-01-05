import logging
import os

import click
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

import measures.algorithms as algos
import mne
from config import LABELED_ROOT
from data.data_files import CHANNEL_NAMES, DataKind, files_builder
from lib.nolitsa.nolitsa import surrogates


@algos.log_result
def compute_sigma(x, true_stat, f):
    surr_sers = [surrogates.iaaft(x)[0] for _ in range(19)]
    surr_stats = [f(surr) for surr in surr_sers]
    return np.abs(np.mean(surr_stats)-true_stat) / np.std(surr_stats), surr_stats


def create_sigma_pkl(in_df, kind, output_path):
    cols = pd.MultiIndex.from_product([CHANNEL_NAMES, algos.measure_names],
                                      names=['channel', 'measure'])
    idxs = pd.MultiIndex.from_product([list(range(1, 134)), ['a', 'b']],
                                      names=['patient', 'trial'])
    main_df = pd.DataFrame(columns=cols, index=idxs)
    for measure_name in in_df.columns.levels[1]:
        for algo in algos.registered_algos:
            if measure_name == algo.algo_name:
                func = algo
                break
        else:
            logging.warning(
                f'Algorithm for measure {measure_name} not regitered, skipping')
            continue

        for row in in_df.iterrows():
            index = row[0][0]
            trial = row[0][1]
            file = files_builder(DataKind(kind)).from_index_trial(index, trial)
            assert((file.id, file.trial) == (index, trial))
            for channel_name in CHANNEL_NAMES:
                true_stat = in_df.loc[(file.id, file.trial), channel_name]
                try:
                    time_series = file.df.loc[:, channel_name][:15000]
                except IndexError:
                    logging.info('Caught index error, skipping')
                    break
                sigma = compute_sigma(time_series, true_stat, func)
                new_row = {(channel_name, measure_name): sigma}
                main_df.loc[(file.id, file.trial)] = pd.Series(new_row)
                logging.debug("New row: \n%s" % new_row)
                logging.debug(f'Saving training data at {output_path}.')
                logging.info(f'Processed file {file.number}')
                main_df.to_pickle(output_path)


@click.command()
@click.argument('in_pkl', type=click.Path(exists=True), default=None)
@click.option('--kind', type=str, default='processed')
def main(in_pkl, kind):
    logging.basicConfig(level=logging.DEBUG)
    mne.set_log_level(logging.ERROR)

    in_df = pd.read_pickle(in_pkl)
    output_path = os.path.join(
        LABELED_ROOT, os.path.splitext(os.path.split(in_pkl)[1])[0]) + \
        '_sigmas.pkl'
    logging.info(f'Creating sigmas for file {in_pkl} in file {output_path}')

    create_sigma_pkl(in_df, kind, output_path)

    logging.info('Finished procedure.')


if __name__ == '__main__':
    main()
