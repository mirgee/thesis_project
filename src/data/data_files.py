import logging
import os
import random
import types
from collections import namedtuple
from enum import Enum, auto

import numpy as np
import pandas as pd

import mne
from config import PROJ_ROOT

DATA_ROOT = os.path.abspath(os.path.join(PROJ_ROOT, 'data'))
CHANNEL_NAMES = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']


def df_from_tdt(file_path):
    return pd.read_table(
        file_path, sep='\t', names=CHANNEL_NAMES, skiprows=[0])


def df_from_fif(file_path):
    raw_fif = mne.io.read_raw_fif(file_path)
    t = pd.DataFrame(raw_fif.get_data())
    return pd.DataFrame(np.transpose(t.values), columns=CHANNEL_NAMES)


def get_meta_df():
    raw_root = os.path.abspath(os.path.join(DATA_ROOT, 'raw'))
    meta_column_names = ['freq', 'RESP_4W', 'RESP_FIN', 'REMISE_FIN', 'AGE',
                         'SEX', 'M_1', 'M_4', 'M_F', 'delka lecby', 'lek 1',
                         'lek 2', 'lek 3', 'lek 4']
    meta_file_name = 'DEP-POOL_Final_144.xlsx'
    return pd.read_excel(
        os.path.join(raw_root, meta_file_name), index_col='ID',
        names=meta_column_names)


def mne_from_file(file):
    sfreq = float(get_meta_df().loc[file.id, 'freq'])
    info = mne.create_info(ch_names=CHANNEL_NAMES, sfreq=sfreq, ch_types='eeg')
    return mne.io.RawArray(np.transpose(file.df.values), info)


class DataKindDefinition:

    def __init__(self, name='', data_folder='', exp_exts=(), df_from_path=None):
        self.name = name
        self.data_folder = data_folder
        self.exp_exts = exp_exts
        self.df_from_path = df_from_path


class DataKind(Enum):
    META = 'meta'
    RAW = 'raw'
    PROCESSED = 'processed'
    MNE = 'mne'
    SURROGATE = 'surrogate'


DATA_KINDS = {
    DataKind.RAW: DataKindDefinition(
        name='raw',
        data_folder=os.path.abspath(os.path.join(DATA_ROOT, 'raw')),
        exp_exts=('.tdt',),
        df_from_path=df_from_tdt),
    DataKind.PROCESSED: DataKindDefinition(
        name='processed',
        data_folder=os.path.abspath(os.path.join(DATA_ROOT, 'processed')),
        exp_exts=('.fif',),
        df_from_path=df_from_fif),
    DataKind.SURROGATE: DataKindDefinition(
        name='surrogate',
        data_folder=os.path.abspath(os.path.join(DATA_ROOT, 'surrogate')),
        exp_exts=('.csv',),
        df_from_path=df_from_tdt),
}


File = namedtuple('File', 'df path id trial name kind')


def files_builder(kind=None, ext=None, file=None, *args, **kwargs):
    def kind_from_extension(ext):
        for kind, definition in DATA_KINDS.items():
            if ext in definition.exp_exts:
                return kind
        raise NotImplementedError(f'File extension {ext} not supported.')

    if ext is not None and kind is None:
        kind = kind_from_extension(ext)
    if kind in DATA_KINDS:
        return DataFiles(DATA_KINDS[kind])
    elif kind == DataKind.META:
        return get_meta_df()
    elif kind == DataKind.MNE:
        return mne_from_file(file)
    else:
        raise NotImplementedError


class DataFiles:

    def __init__(self, kind, shuffle=False):
        self.kind = kind.name
        self.exp_exts = kind.exp_exts
        self.data_folder = kind.data_folder
        self.df_from_path = kind.df_from_path
        self.shuffle = shuffle

    def file_names(self):
        file_names = os.listdir(self.data_folder)
        if self.shuffle:
            random.shuffle(file_names)
        for file_name in file_names:
            _, ext = os.path.splitext(file_name)
            if ext not in self.exp_exts:
                logging.debug(
                    f'Unexpected extension: skipping file {file_name}.')
                continue
            yield file_name

    def single_file(self, file_name):
        _, ext = os.path.splitext(file_name)
        assert (file_name in os.listdir(self.data_folder)
                and ext in self.exp_exts)
        file_path = os.path.join(self.data_folder, file_name)
        return File(
            df=self.df_from_path(file_path),
            id=self.get_index(file_path),
            trial=self.get_trial(file_path),
            path=file_path,
            name=file_name,
            kind=self.kind)

    def get_index(self, file_name):
        no_ext_file_name = os.path.splitext(file_name)[0]
        return int(no_ext_file_name[:-1])

    def get_trial(self, file_name):
        no_ext_file_name = os.path.splitext(file_name)[0]
        return no_ext_file_name[-1]

    def __iter__(self):
        for file_name in self.file_names():
            file_path = os.path.join(self.data_folder, file_name)
            yield File(
                df=self.df_from_path(file_path),
                id=self.get_index(file_name),
                trial=self.get_trial(file_name),
                path=file_path,
                name=file_name,
                kind=self.kind)
