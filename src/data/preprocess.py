import click
import pandas as pd

from ..data.utils import read_raw


def get_featues(dframe):
    return dframe[['x0', 'x1', 'x2', 'x3']]


def get_label(dframe):
    return dframe['y']


def preprocess_data(dframe):
    dframe = dframe.copy()  # I want to avoid inplace modifications
    dframe.columns = ['x0', 'x1', 'x2', 'x3', 'y']
    return dframe


def read_processed_data(fname='data/processed/processed.pickle'):
    dframe = pd.read_pickle(fname)
    return dframe


@click.command()
@click.argument('input_file', type=click.Path(exists=True, readable=True, dir_okay=False))
@click.argument('output_file', type=click.Path(writable=True, dir_okay=False))
@click.option('--excel', type=click.Path(writable=True, dir_okay=False))
def main(input_file, output_file, excel):
    print('Preprocessing data')

    dframe = read_raw_data(input_file)
    dframe = preprocess_data(dframe)

    dframe.to_pickle(output_file)
    if excel is not None:
        dframe.to_excel(excel)


if __name__ == '__main__':
    main()
