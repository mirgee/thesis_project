import logging

import click
import pandas as pd


def concatenate_dfs(inf, outf):
    dfs = [pd.read_pickle(infile) for infile in inf]
    new = pd.concat(dfs, axis=1)
    logging.info(f'Resulting dataframe:\n{new}')
    new.to_pickle(outf)


@click.command()
@click.argument('inf', type=click.Path(exists=True, readable=True,
                                       dir_okay=False), nargs=-1)
@click.argument('outf', type=click.Path(exists=False, readable=False,
                                        dir_okay=False), nargs=1)
def main(inf, outf):
    logging.basicConfig(level=logging.INFO)

    logging.info('Concatenating dataframes...')

    concatenate_dfs(inf, outf)


if __name__ == '__main__':
    main()
