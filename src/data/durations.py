import logging
import os

import pandas as pd

from config import DURATIONS_ROOT, RAW_ROOT
from data_files import DataKind, files_builder
from utils import get_duration, get_sampling_frequency, get_trial, get_index


def durations(input_path=RAW_ROOT, output_file=DURATIONS_ROOT,
              data_kind=DataKind('raw')):
    durs_path = os.path.join(output_file, 'durations.pkl')
    df_durs = pd.read_csv(durs_path, sep='\t') if os.path.isfile(durs_path) \
        else _save_durations_csv(input_path, durs_path, data_kind)
    _summarize(df_durs, output_file)


def _save_durations_csv(input_path, durs_path, data_kind):
    it_multi = pd.MultiIndex.from_product([list(range(1, 134)), ['a', 'b']],
                                          names=['patient', 'trial'])
    df_durs = pd.DataFrame(index=it_multi, columns=['duration_s', 'sfreq'])
    for file in files_builder(data_kind):
        file_path = os.path.join(input_path, file.name)
        index = get_index(file_path)
        trial = get_trial(file_path)
        df_durs.loc[(index, trial), 'duration_s'] = get_duration(file_path, file.df)
        df_durs.loc[(index, trial), 'sfreq'] = float(get_sampling_frequency(file_path))
    df_durs.to_pickle(durs_path)
    return df_durs


def _summarize(df_durs, output_file):
    with open(os.path.join(output_file, 'summary'), 'w') as f:
        f.write(str(df_durs.describe()))
        f.write('\n\n')
        f.write(str(df_durs.groupby('sfreq').describe()))


def main():
    logging.basicConfig(level=logging.INFO)
    logging.info('Saving recording durations...')

    durations()


if __name__ == '__main__':
    main()
