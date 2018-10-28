import logging
import os

import pandas as pd

import mne
from config import CHANNEL_NAMES, LABELED_ROOT, PROCESSED_ROOT
from data.utils import df_from_fif, get_trial_index, get_trials
from labels.non_linear import (FEATURE_NAMES, compute_corr_dim, compute_dfa,
                               compute_hurst, compute_lyapunov)


def _features_for_channel_chunk(trial, features, chunk, chunk_num):
    row = {}
    # TODO: Make a mapping of {feat_name: function} and just loop through it
    if 'lyap' in features:
        row.update({('lyap', chunk_num): compute_lyapunov(chunk)})
    if 'corr' in features:
        row.update({('corr', chunk_num): compute_corr_dim(chunk)})
    if 'dfa' in features:
        row.update({('dfa', chunk_num): compute_dfa(chunk)})
    if 'hurst' in features:
        row.update({('hurst', chunk_num): compute_hurst(chunk)})

    return row


def _create_df(input_path, feature_names):
    trials = get_trials(input_path)
    index = pd.MultiIndex.from_product([trials, CHANNEL_NAMES],
                                       names=['trial', 'channel'])
    cols = pd.MultiIndex.from_product([feature_names,
                                       range(len(feature_names)+1)],
                                      names=['feature', 'chunk'])
    return pd.DataFrame(columns=cols, index=index)


def compute_nl_split(splits, feature_names, trial, df_in, df_out,
                     compute_overall=False):
    for channel in CHANNEL_NAMES:
        values = df_in[channel].values
        n = len(values)
        new_row = {}
        for split in range(splits):
            chunk = values[split*(n//splits):(split+1)*(n//splits)]
            new_row.update(_features_for_channel_chunk(
                trial, feature_names, chunk, split))
        if compute_overall:
            new_row.update(
                _features_for_channel_chunk(
                    trial, feature_names, values, len(splits)))
        df_out.loc[(trial, channel)] = pd.Series(new_row)
        logging.debug("Adding row: \n %s" % df_out.loc[(trial, channel)])
    return df_out


def create_split_data(feature_names=FEATURE_NAMES, input_path=PROCESSED_ROOT,
                      output_path=LABELED_ROOT):
    """Create a dataframe with features and labels made up of signals splitted
    and NL features computed per sub-signal."""
    logging.info('Creating splitted nl coeffs data...')
    df = _create_df(input_path, feature_names)

    for file_name in os.listdir(input_path):
        if not file_name.endswith('.fif'):
            logging.info('Skipping file %s' % file_name)
            continue

        file_path = os.path.join(input_path, file_name)
        _, _, trial = get_trial_index(file_name)
        try:
            df_in = df_from_fif(file_path, 60)
        except Exception:
            duration = get_duration(file_path)
            logging.info(f'Skipping file {file_name} with duration {duration}'
                         f' s.')
            continue

        df = compute_nl_split(4, feature_names, trial, df_in, df)

    logging.info('Saving training data as pickle...')
    df.to_pickle(os.path.join(output_path, 'splits.pickle'))


def main():
    logging.basicConfig(level=logging.DEBUG)
    mne.set_log_level(logging.WARNING)

    create_split_data()


if __name__ == '__main__':
    main()
