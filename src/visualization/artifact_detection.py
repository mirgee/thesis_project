import logging
import os
import random as rd

import click
import mne
import pandas as pd
import numpy as np

from mne.datasets import sample

from mne.preprocessing import ICA
from mne.preprocessing import create_eog_epochs, create_ecg_epochs
from mne.preprocessing.ica import corrmap

from config import CHANNEL_NAMES, DATA_ROOT, PROCESSED_ROOT, RAW_ROOT
from data.utils import df_from_fif, df_from_tdt, get_sfreq, raw_mne_from_tdt


def detect_eog_from_info(input_file):
    raw = raw_mne_from_tdt(input_file)

    #TODO: this would work if we had some info about eog or meg events in the
    # raw info file...
    picks_eog = mne.pick_types(raw.info, meg=false, eeg=false, eog=true,
                               stim=false, exclude='bads')
    reject = dict(mag=5e-12, grad=4000e-13, eeg=80e-6)
    # lowpass at 50
    raw.filter(None, 50., fir_design='firwin')
    # 1hz high pass is often helpful for fitting ica
    raw.filter(1., None, n_jobs=1, fir_design='firwin')

    eog_average = create_eog_epochs(raw, reject=reject, picks=picks_eog).average()

    eog_epochs = create_eog_epochs(raw, reject=reject)  # get single EOG trials
    eog_inds, scores = ica.find_bads_eog(eog_epochs)  # find via correlation

    ica.plot_scores(scores, exclude=eog_inds)  # look at r scores of components

    ica.plot_sources(eog_average, exclude=eog_inds)  # look at source time course


def run_ica(raw):
    method = 'fastica'

    # Choose other parameters
    n_components = 25  # if float, select n_components by explained variance of PCA
    decim = 3  # we need sufficient statistics, not all time points -> saves time

    # we will also set state of the random number generator - ICA is a
    # non-deterministic algorithm, but we want to have the same decomposition
    # and the same order of components each time this tutorial is run
    random_state = 23

    # reject = dict(mag=5e-12, grad=4000e-13)
    reject = None
    ica = ICA(n_components=None, method=method, random_state=random_state)

    ica.fit(raw, decim=decim, reject=reject)
    
    return ica


def find_template_manually_ica(input_file):
    raw = raw_mne_from_tdt(input_file)

    # lowpass at 50
    raw.filter(None, 50., fir_design='firwin')
    # 1hz high pass is often helpful for fitting ica
    raw.filter(1., None, n_jobs=1, fir_design='firwin')

    ica = run_ica(raw)

    # TODO: To make the plot, we would need set locations for the electrodes
    # manually, because it is not in the data
    ica.plot_components()


def detect_all():
    input_folder = RAW_ROOT

    to_examine = os.listdir(input_folder)
    rd.shuffle(to_examine)

    for file_name in to_examine:
        # detect_eog(os.path.join(input_folder, file_name))
        find_template_manually_ica(os.path.join(input_folder, file_name))


@click.command()
def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    detect_all()

if __name__ == '__main__':
    main()
