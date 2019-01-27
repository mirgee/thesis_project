import glob
import logging
import os
import random
import types
from collections import namedtuple
from enum import Enum, auto

import numpy as np
import pandas as pd

import mne
from config import PROJ_ROOT, LABELED_ROOT

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


def data_from_npy(file_path):
    return np.load(file_path)


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
    RECPLOT = 'recplot'
    GAF = 'gaf'


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
    DataKind.RECPLOT: DataKindDefinition(
        name='recplot',
        data_folder=os.path.abspath(os.path.join(DATA_ROOT, 'recplots')),
        exp_exts=('.npy',),
        df_from_path=data_from_npy),
    DataKind.GAF: DataKindDefinition(
        name='gaf',
        data_folder=os.path.abspath(os.path.join(DATA_ROOT, 'gaf')),
        exp_exts=('.npy',),
        df_from_path=data_from_npy),
}


File = namedtuple('File', 'df path id trial name kind number')


def files_builder(kind=None, ext=None, file=None, subfolder=(), *args, **kwargs):
    def kind_from_extension(ext):
        for kind, definition in DATA_KINDS.items():
            if ext in definition.exp_exts:
                return kind
        raise NotImplementedError(f'File extension {ext} not supported.')

    if ext is not None and kind is None:
        kind = kind_from_extension(ext)
    if kind in DATA_KINDS:
        return DataFiles(DATA_KINDS[kind], subfolder=subfolder)
    elif kind == DataKind.META:
        return get_meta_df()
    elif kind == DataKind.MNE:
        return mne_from_file(file)
    else:
        raise NotImplementedError


class DataFiles:

    def __init__(self, kind, shuffle=False, subfolder=()):
        assert os.path.isdir(kind.data_folder), kind.data_folder
        self.kind = kind.name
        self.exp_exts = kind.exp_exts
        self.data_folder = kind.data_folder
        if len(subfolder) > 0:
            self.data_folder = os.path.join(*((kind.data_folder,) + subfolder))
        self.df_from_path = kind.df_from_path
        self.shuffle = shuffle
        self.numfiles = len(os.listdir(self.data_folder))

    def file_names(self, include_path=False, subfolder=(), recursive=False):
        data_folder = os.path.join(*((self.data_folder,) + subfolder))
        if recursive:
            file_names = glob.glob(data_folder + '/**/*'+self.exp_exts[0], recursive=True)
        else:
            file_names = os.listdir(data_folder)
        if include_path and not recursive:
            file_names = [os.path.join(data_folder, fn) for fn in file_names]
        if self.shuffle:
            random.shuffle(file_names)
        for i, file_name in enumerate(file_names):
            _, ext = os.path.splitext(file_name)
            if ext not in self.exp_exts:
                logging.debug(
                    f'Unexpected extension: skipping file {file_name}.')
                continue
            yield i, file_name

    def train_test_file_names(self, test_size=0.3):
        assert test_size < 1, 'test_size must be < 1'
        all_names = [os.path.join(self.data_folder, name[1])
                     for name in self.file_names()]
        return all_names[int(test_size*len(all_names)):], \
                all_names[:int(test_size*len(all_names))]

    def get_labels(self, file_names=None, label='dep'):
        if file_names is None:
            file_names = [fn for _, fn in self.file_names()]
        ls = pd.read_pickle(
            os.path.join(LABELED_ROOT, 'processed', 'meta', 'meta.pkl'))
        return [
            ls.loc[(self.get_index(fn), self.get_trial(fn)), label]
            for fn in file_names
        ]

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
            kind=self.kind,
            number=None)

    def from_index_trial(self, index, trial):
        file_name = ''.join((str(index), str(trial))) + self.exp_exts[0]
        assert (file_name in os.listdir(self.data_folder)), file_name
        file_path = os.path.join(self.data_folder, file_name)
        return File(
            df=self.df_from_path(file_path),
            id=index,
            trial=trial,
            path=file_path,
            name=file_name,
            kind=self.kind,
            number=None)

    def get_index(self, file_name):
        no_ext_file_name = os.path.split(os.path.splitext(file_name)[0])[1]
        i = no_ext_file_name.find('-')
        if i > 0:
            no_ext_file_name = no_ext_file_name[:i]
        return int(no_ext_file_name[:-1])

    def get_trial(self, file_name):
        no_ext_file_name = os.path.split(os.path.splitext(file_name)[0])[1]
        i = no_ext_file_name.find('-')
        if i > 0:
            no_ext_file_name = no_ext_file_name[:i]
        return no_ext_file_name[-1]

    def __iter__(self):
        for i, file_name in self.file_names():
            file_path = os.path.join(self.data_folder, file_name)
            yield File(
                df=self.df_from_path(file_path),
                id=self.get_index(file_name),
                trial=self.get_trial(file_name),
                path=file_path,
                name=file_name,
                kind=self.kind,
                number=f'{i}/{self.numfiles}')
