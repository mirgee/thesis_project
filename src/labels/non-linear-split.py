import nolds
import pandas as pd
import logging
import os
import numpy as np

from data.utils import df_from_fif, get_trial_index, get_meta_df
from config import CHANNEL_NAMES, PROCESSED_ROOT, LABELED_ROOT
from lib.nolitsa.dimension import fnn
from labels.non-linear import compute_lyapunov, compute_corr_dim, compute_dfa,
    compute_hurst, compute_label, FEATURES

def compute_nl_split(file_path, splits):
    df = df_from_fif(file_path)
    features = {
        'lyap': [],
        'corr': [],
        'dfa': [],
        'hurst': [],
    }

    for channel in CHANNEL_NAMES:
        values = df[channel].values
        n = len(values)
        for split in range(splits):
            chunk = values[split*(n//splits):(split+1)*(n//splits)]

            features['lyap'].append(compute_lyapunov(chunk))
            features['corr'].append(compute_corr_dim(chunk))
            features['dfa'].append(compute_dfa(chunk))
            features['hurst'].append(compute_hurst(chunk))

        features['lyap'].append(compute_lyapunov(values))
        features['corr'].append(compute_corr_dim(values))
        features['dfa'].append(compute_dfa(values))
        features['hurst'].append(compute_hurst(values))

    return features

def create_split_data(input_path=PROCESSED_ROOT, output_path=LABELED_ROOT):
    """Create a dataframe with features and labels made up of signals splitted
    and NL features computed per sub-signal."""
    logging.info('Creating splitted nl coeffs data...')

    id_name = 'ID'
    cols = [id_name] + ['-'.join((f,c)) for c in CHANNEL_NAMES for f in \
                     FEATURES] + ['label']

    dfs = {
        'lyap': pd.DataFrame(columns=cols, index=id_name),
        'corr': pd.DataFrame(columns=cols, index=id_name),
        'dfa': pd.DataFrame(columns=cols, index=id_name),
        'hurst': pd.DataFrame(columns=cols, index=id_name),
    }

    for file_name in os.listdir(input_path):

        if not file_name.endswith('.fif'):
            logging.info('Skipping file %s' % file_name)
            continue

        file_path = os.path.join(input_path, file_name)

        features = compute_nl_split(file_path, splits=4)
        label = compute_label(file_path)

        trial, idx = get_trial_index(file_name)
        for feature, df in dfs.items():
            new_df = pd.DataFrame([idx] + features[feature] + [label],
                                  columns=cols, index=id_name)
            df = df.append(new_df)

    logging.info('Saving training data as pickle...')
    for feature, df in dfs.items():
        pickle_name = feature + '.pickle'
        df.to_pickle(os.path.join(output_path, pickle_name))


def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    create_training_data()

if __name__ == '__main__':
    main()
