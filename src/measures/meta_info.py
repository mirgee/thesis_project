import logging
import os

import pandas as pd
import click

from config import LABELED_ROOT
from data.data_files import DataKind, files_builder


def create_meta_df(output_path):
    logging.info('Creating dataframe with the meta information.')
    cols = ['resp', 'b/a', 'sex', 'age', 'sc', 'sc_bef', 'sc_aft', 'change']
    idxs = pd.MultiIndex.from_product([list(range(1, 134)), ['a', 'b']],
                                      names=['patient', 'trial'])
    extra_df = pd.DataFrame(columns=cols, index=idxs)
    meta_df = files_builder(DataKind.META)
    for file in files_builder(DataKind.PROCESSED):
        index = file.id
        trial = file.trial
        meta_row = meta_df.loc[index, :]
        extra_df.loc[(index, trial)]['resp'] = meta_row['RESP_4W']
        extra_df.loc[(index, trial)]['b/a'] = 0 if trial == 'a' else 1
        extra_df.loc[(index, trial)]['age'] = meta_row['AGE']
        extra_df.loc[(index, trial)]['sex'] = meta_row['SEX']
        m1 = meta_row['M_1']
        m4 = meta_row['M_4']
        extra_df.loc[(index, trial)]['sc'] = m1 if trial == 'a' else m4
        extra_df.loc[(index, trial)]['sc_bef'] = m1
        extra_df.loc[(index, trial)]['sc_aft'] = m4
        extra_df.loc[(index, trial)]['change'] = m1/m4
        logging.debug('Added row: \n{}'.format(
            extra_df.loc[(index, trial)]))

    extra_df = extra_df.astype(
        {'resp': 'category',
         'b/a': 'category',
         'sex': 'category',
         'age': int,
         'sc': float,
         'sc_bef': int,
         'sc_aft': int,
         'change': float})
    logging.debug('The resulting data: \n{}'.format(
        extra_df.describe()))
    logging.debug(f'Saving metadata dataframe at {output_path}.')
    extra_df.to_pickle(output_path)

    output_folder = os.path.sep.join(output_path.split(os.sep)[:-1])
    measures_file = os.path.join(LABELED_ROOT, 'all', 'training.pickle')
    if os.path.isfile(measures_file):
        measures_df = pd.read_pickle(measures_file)
        joined_df = measures_df.join(extra_df)
        joined_path = os.path.join(output_folder, 'measures_w_meta.pkl')
        logging.debug(f'Saving joined dataframe at {joined_path}:\n'
                      f'{joined_df}')
        joined_df.to_pickle(joined_path)


@click.command()
@click.option('--out', type=str, default='meta.pkl')
def main(out):
    logging.basicConfig(level=logging.DEBUG)

    output_path = os.path.join(LABELED_ROOT, 'meta', out)
    create_meta_df(output_path)


if __name__ == '__main__':
    main()
