import pandas as pd
import numpy as np
import logging
import os
import click

from data.utils import raw_mne_from_tdt, remove_extension

from config import RAW_ROOT, PROCESSED_ROOT


def remove_outliers(data, alpha):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    attempts = 0
    while attempts < 100:
        alpha = 0.1
        outliers = [x for x in data if mean-alpha*std < x < mean+alpha*std]

        percent = len(outliers) / len(data)

        if percent < 0.07:
            data = np.array([x for x,i in enumerate(data) if x not in outliers or
                             i==0])
            break
        else:
            alpha *= 1.5
            attempts += 1
    else:
        raise Exception('Failed to eliminate outliers')

    return data


def preprocess_raw_mne_file(mne_raw_data, proj=False):
    if proj:
        # Apply averaging projection (remove outliers)
        mne_raw_data.set_eeg_reference('average', projection=True)
        mne_raw_data.apply_proj()

    # Remove power line noise, may not be necessary depending on the data
    # mne_raw_data.notch_filter(np.arange(50, 125, 50), filter_length='auto', phase='zero')
    mne_raw_data.notch_filter(50, filter_length='auto', phase='zero')

    # Or we can simply low pass filter, which may be better
    mne_raw_data.filter(.5, None, fir_design='firwin')
    mne_raw_data.filter(None, 70., fir_design='firwin')

    # Remove slow drifts via high pass, bad practice in some situations, but may improve ICA
    # mne_raw_data.filter(1., None, fir_design='firwin')

    # Downsample high frequency recordings to reduce the amount of data
    if int(mne_raw_data.info['sfreq']) == 1000:
        logging.info('Downsampling data...')
        mne_raw_data.resample(250, npad="auto")

    return mne_raw_data


def preprocess_all(input_path=RAW_ROOT, output_file=PROCESSED_ROOT):
    for file_name in os.listdir(input_path):
        if not file_name.endswith('.tdt'):
            logging.info('Skipping file %s' % file_name)
            continue
        file_path = os.path.join(input_path, file_name)
        mne_raw_data = raw_mne_from_tdt(file_path)

        proj = False
        mne_raw_data = preprocess_raw_mne_file(mne_raw_data, proj)

        processed_file_name = remove_extension(file_name) + '.fif'
        mne_raw_data.save(os.path.join(output_file, processed_file_name),
                          proj=proj, overwrite=True)

    return mne_raw_data


@click.command()
@click.option('--input_folder', type=click.Path(exists=True, readable=True, dir_okay=False))
@click.option('--output_folder', type=click.Path(writable=True, dir_okay=False))
def main(input_folder, output_folder):
    logging.basicConfig(level=logging.INFO)

    logging.info('Preprocessing data')

    preprocess_all()


if __name__ == '__main__':
    main()
