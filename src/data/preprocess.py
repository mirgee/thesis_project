import pandas as pd
import numpy as np
import logging
import os
import click


from config import PROCESSED_ROOT
from data.data_files import DataKind, files_builder


def preprocess_raw_mne_file(mne_raw_data, proj=False):
    if proj:
        # Apply averaging projection (remove outliers)
        mne_raw_data.set_eeg_reference('average', projection=True)
        mne_raw_data.apply_proj()

    # Remove power line noise, may not be necessary depending on the data
    # mne_raw_data.notch_filter(np.arange(50, 125, 50), filter_length='auto', phase='zero')
    # mne_raw_data.notch_filter(50, filter_length='auto', phase='zero')

    # Or we can simply low pass filter, which may be better
    # mne_raw_data.filter(.5, None, fir_design='firwin')
    # mne_raw_data.filter(None, 70., fir_design='firwin')

    # Remove slow drifts via high pass, bad practice in some situations, but may improve ICA
    # mne_raw_data.filter(1., None, fir_design='firwin')

    # Downsample high frequency recordings to reduce the amount of data
    if int(mne_raw_data.info['sfreq']) == 1000:
        logging.info('Downsampling data...')
        mne_raw_data.resample(250, npad="auto")

    # mne_raw_data.crop(0, 60)

    return mne_raw_data


def preprocess_all(output_file=PROCESSED_ROOT):
    for file in files_builder(DataKind.RAW):
        mne_raw_data = files_builder(DataKind.MNE, file=file)

        try:
            mne_raw_data = preprocess_raw_mne_file(mne_raw_data)
        except ValueError:
            # Raised when duration is < 60 s, we may safely skip the file
            logging.debug(f'Skipping file {file.name} because of insufficient '
                          'duration.')
            continue

        processed_file_name = os.path.splitext(file.name)[0] + '.fif'
        mne_raw_data.save(os.path.join(output_file, processed_file_name),
                          proj=False, overwrite=True)

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
