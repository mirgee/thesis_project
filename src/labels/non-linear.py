import nolds
import pandas as pd
import logging

from utils import df_from_fif
from config import CHANNEL_NAMES

FEATURES = ['lyap', 'corr', 'dfa', 'higu']

def compute_lyapunov(data):
    pass


def compute_corr_dim(data):
    pass


def compute_dfa(data):
    pass


def compute_higuchi(data):
    pass


def compute_nl(file_path):
    """Compute dict of non-linear features for trial recorded in file_name"""
    df = df_from_fif(file_name)
    features = {}

    for channel in CHANNEL_NAMES:
        lyap = compute_lyapunov(df[channel])
        corr = compute_corr_dim(df[channel])
        dfa = compute_dfa(df[channel])
        higu = compute_higuchi(df[channel])

        features[channel] = [lyap, corr, dfa, higu]

    return features


def compute_label(file_path):
    #TODO
    pass


def create_training_data(input_path=PROCESSED_ROOT, output_path=LABELED_ROOT):
    """Create a dataframe with features and labels suitable for training."""
    logging.info('Creating training data...')

    cols = ['-'.join((f,c)) for c in CHANNEL_NAMES for f in FEATURES] + ['label']
    main_df = pd.DataFrame(columns=cols)

    for file_name in os.listdir(input_path):
        if not file_name.endswith('.fif'):
            logging.info('Skipping file %s' % file_name)
            continue
        file_path = os.path.join(input_path, file_name)

        features = compute_nl(file_path)
        label = compute_label(file_path)

        df = pd.DataFrame(features + [label], columns=cols)
        main_df = main_df.append(df, ignore_index=True)

    logging.info('Saving training data as pickle...')
    main_df.to_pickle(output_path)


@click.command()
def main():
    pass

if __name__ == '__main__':
    main()
