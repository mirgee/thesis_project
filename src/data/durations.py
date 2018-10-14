import logging
import pandas as pd
import os

from config import RAW_ROOT, DURATIONS_ROOT
from utils import (df_from_tdt, get_sampling_frequency, get_trials,
                   get_trial_index)


def save_durations(input_path=RAW_ROOT, output_file=DURATIONS_ROOT):
    trials = get_trials(input_path)
    df_durs = pd.DataFrame(index=trials, columns=['duration_s', 'sfreq'],
                           dtype=float)
    for file_name in os.listdir(input_path):
        if not file_name.endswith('.tdt'):
            logging.info('Skipping file %s' % file_name)
            continue
        file_path = os.path.join(input_path, file_name)
        df_rec = df_from_tdt(file_path)
        num_rows = len(df_rec.index)
        sfreq = get_sampling_frequency(file_path)
        _, _, trial = get_trial_index(file_path)
        df_durs.loc[trial, 'duration_s'] = (num_rows-1)/sfreq
        df_durs.loc[trial, 'sfreq'] = float(sfreq)
    df_durs.sort_index().to_csv(
        os.path.join(output_file, 'durations.csv'), sep='\t')
    with open(os.path.join(output_file, 'summary'), 'w') as f:
        f.write(str(df_durs.describe()))
        f.write('\n\n')
        f.write(str(df_durs.groupby('sfreq').describe()))


def main():
    logging.basicConfig(level=logging.INFO)
    logging.info('Saving recording durations...')

    save_durations()


if __name__ == '__main__':
    main()
