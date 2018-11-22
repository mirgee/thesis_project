import logging
import os

import pandas as pd

from config import SURROGATE_ROOT
from data.data_files import DataKind, files_builder
from lib.nolitsa.nolitsa.surrogates import iaaft


def create_surrogates():
    for file in files_builder(DataKind.PROCESSED):
        surr_df = pd.DataFrame().reindex_like(file.df)
        for col in file.df.columns:
            surr_df[col] = iaaft(
                file.df[col], maxiter=1000, atol=1e-8, rtol=1e-10)[0]
        file_name = os.path.splitext(file.name)[0] + '.csv'
        file_path = os.path.join(SURROGATE_ROOT, file_name)
        surr_df.to_csv(file_path, sep='\t')


def main():
    logging.basicConfig(level=logging.INFO)

    logging.info('Creating surrogate data...')

    create_surrogates()


if __name__ == '__main__':
    main()
