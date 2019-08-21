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
from config import LABELED_ROOT, PROJ_ROOT, DATA_ROOT, CHANNEL_NAMES
from utils import (get_index, get_trial, df_from_tdt, df_from_fif,
                   data_from_npy, get_meta_df, mne_from_file)


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
    DIRECT = 'direct'


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
    DataKind.DIRECT: DataKindDefinition(
        name='direct',
        data_folder=os.path.abspath(os.path.join(DATA_ROOT, 'direct')),
        exp_exts=('.npy',),
        df_from_path=data_from_npy),
}


File = namedtuple('File', 'df path id trial name kind number')


def files_builder(kind=None, ext=None, file=None, subfolder=(), *args, **kwargs):
    """Creates a DataFiles iterator based on kind, extension, or returns a
    single dataframe based on file."""
    def kind_from_extension(ext):
        """Selects the first datakind matching the provided extension."""
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
    """
    Iterator over file names of supplied properties. It supports shuffling,
    subfolders, absolute / relative paths, and selection only before / after
    trials."""

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

    def file_names(self, include_path=False, subfolder=(), recursive=False,
                   index_trials=None):
        """Generator of file names."""
        data_folder = os.path.join(*((self.data_folder,) + subfolder))
        if recursive:
            file_names = glob.glob(data_folder + '/**/*'+self.exp_exts[0], recursive=True)
        else:
            file_names = os.listdir(data_folder)
        if index_trials is not None:
            file_names = [fn for fn in file_names
                          if fn.split('-')[0] in index_trials]
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
        """Split the file names into train / test samples."""
        assert test_size < 1, 'test_size must be < 1'
        all_names = [os.path.join(self.data_folder, name[1])
                     for name in self.file_names()]
        return all_names[int(test_size*len(all_names)):], \
            all_names[:int(test_size*len(all_names))]

    def get_filenames_with_labels(self, file_names=None, label='dep',
                                  trial=None):
        """Get only labels to supplied filenames or all filenames with
        corresponding labels."""
        if file_names is None:
            file_names = [fn for _, fn in self.file_names(include_path=True)]
        ls = pd.read_pickle(
            os.path.join(LABELED_ROOT, 'processed', 'meta', 'meta.pkl'))
        if trial is None:
            file_names, labels = file_names, [
                ls.loc[(self.get_index(fn), get_trial(fn)), label]
                for fn in file_names
            ]
        else:
            file_names = [fn for fn in file_names if
                          get_trial(fn) == trial]
            labels = [
                ls.loc[(self.get_index(fn), trial), label]
                for fn in file_names
            ]
        return file_names, labels

    def single_file(self, file_name):
        """Get file instance corresponding to file of supplied name."""
        _, ext = os.path.splitext(file_name)
        assert (file_name in os.listdir(self.data_folder)
                and ext in self.exp_exts)
        file_path = os.path.join(self.data_folder, file_name)
        return File(
            df=self.df_from_path(file_path),
            id=get_index(file_path),
            trial=get_trial(file_path),
            path=file_path,
            name=file_name,
            kind=self.kind,
            number=None)

    def from_index_trial(self, index, trial):
        """Get File instance of the file corresponding to supplied index and
        trial."""
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

    def __iter__(self):
        for i, file_name in self.file_names():
            file_path = os.path.join(self.data_folder, file_name)
            yield File(
                df=self.df_from_path(file_path),
                id=get_index(file_name),
                trial=get_trial(file_name),
                path=file_path,
                name=file_name,
                kind=self.kind,
                number=f'{i}/{self.numfiles}')
