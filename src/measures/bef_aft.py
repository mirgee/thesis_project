from functools import reduce
import logging
import os

import click
import pandas as pd

import measures.algorithms as algos
from config import CHANNEL_NAMES, LABELED_ROOT


def create_bef_aft_df(output_path, kind, measure='all', format='pkl'):
    df = pd.read_pickle(
        os.path.join(output_path, f'training_{measure}.pkl'))
    logging.debug(f'TRAINING DF FOR {measure.upper()}:\n{df}')
    if isinstance(df.columns, pd.core.index.MultiIndex):
        col = slice(None) if measure == 'all' else measure
        df = df.loc[:, (slice(None), col)].dropna()
        df.columns = df.columns.droplevel(1)
        df = df.astype(float)

    df_y = pd.read_pickle(os.path.join(LABELED_ROOT, kind, 'meta', 'meta.pkl'))
    # if set(df_y.columns).isdisjoint(set(df.columns)):
    if set(df_y.columns) != set(df.columns):
        df = df.drop(df_y.columns, axis=1, errors='ignore')
        df = df.join(df_y)

    df_bef = df.loc[(slice(None), 'a'), :]
    df_bef.index = df_bef.index.droplevel(1)
    df_aft = df.loc[(slice(None), 'b'), :]
    df_aft.index = df_aft.index.droplevel(1)

    in_both = df_bef.join(df_aft, how='inner', lsuffix='_l').index
    df = df.loc[(in_both, slice(None)), :]
    df_bef = df_bef.loc[in_both, :]
    df_aft = df_aft.loc[in_both, :]

    if format == 'csv':
        df_bef.to_csv(
            os.path.join(output_path, f'{measure}_bef.csv'))
        df_aft.to_csv(
            os.path.join(output_path, f'{measure}_aft.csv'))
        if measure is not 'all':
            df.to_csv(
                os.path.join(output_path, f'training_{measure}.csv'))
    elif format == 'pkl':
        df_bef.to_pickle(
            os.path.join(output_path, f'{measure}_bef.pkl'))
        df_aft.to_pickle(
            os.path.join(output_path, f'{measure}_aft.pkl'))
        if measure is not 'all':
            df.to_pickle(
                os.path.join(output_path, f'training_{measure}.pkl'))
    else:
        raise NotImplementedError(f'Format {format} not supported.')

    logging.debug(f'{measure.upper()} ORIGINAL DATAFRAME:\n{df}')
    logging.debug(f'{measure.upper()} PATIENTS BEFORE:\n{df_bef}')
    logging.debug(f'{measure.upper()} PATIENTS AFTER:\n{df_aft}')


def join_to_all(measures, output_path, kind):
    dfs_t1 = [
        pd.read_pickle(
            os.path.join(
                LABELED_ROOT, kind, m, f'training_{m}.pkl')) for m in measures]
    dfs_t2 = [df.loc[:, CHANNEL_NAMES] for df in dfs_t1]
    dfs = [pd.concat(
        [df], keys=[m], axis=1).swaplevel(0, 1, 1) for m, df in zip(
            measures, dfs_t2)]
    df_all = reduce(
        lambda left,right: pd.merge(
            left, right, how='outer', left_index=True, right_index=True), dfs)
    df_all.to_pickle(os.path.join(output_path, 'all', 'training_all.pkl'))
    logging.debug(f'CREATED ALL:\n{df_all}')


@click.command()
@click.argument('measures', nargs=-1, type=str)
@click.option('--format', type=str, default='pkl')
@click.option('--kind', type=str, default='processed')
def main(measures, format, kind):
    logging.basicConfig(level=logging.DEBUG)

    logging.info('Creating dataframe of all input measures.')
    output_path = os.path.join(LABELED_ROOT, kind)
    join_to_all(measures, output_path, kind)
    return

    logging.info('Creating before and after dataframes.')

    for measure in measures:
        output_path = os.path.join(LABELED_ROOT, kind, measure)
        try:
            create_bef_aft_df(output_path, kind, measure, format)
            logging.info(f'Successfully created before and after file for '
                         f'{measure}.')
        except FileNotFoundError:
            logging.info(f'File for measure {measure} not found on path '
                         f'{output_path}, skipping.')


if __name__ == '__main__':
    main()
