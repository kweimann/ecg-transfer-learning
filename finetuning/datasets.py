import functools

import numpy as np

from transplant.datasets import physionet
from transplant.utils import pad_sequences


def get_challenge17_data(db_dir, fs=None, pad=None, normalize=False, verbose=False):
    records, labels = physionet.read_challenge17_data(db_dir, verbose=verbose)
    if normalize:
        normalize = functools.partial(
            physionet.normalize_challenge17, inplace=True)
    data_set = _prepare_data(
        records,
        labels,
        normalize_fn=normalize,
        fs=fs,
        pad=pad,
        verbose=verbose)
    return data_set


def get_challenge20_data(db_dir, fs=None, pad=None, normalize=False, verbose=False):
    records, labels = physionet.read_challenge20_data(db_dir, verbose=verbose)
    if normalize:
        normalize = functools.partial(
            physionet.normalize_challenge20, inplace=True)
    data_set = _prepare_data(
        records,
        labels,
        normalize_fn=normalize,
        fs=fs,
        pad=pad,
        verbose=verbose)
    return data_set


def get_ptb_xl_data(db_dir, fs=None, pad=None, normalize=False,
                    category='rhythm', folds=None, verbose=False):
    records, labels = physionet.read_ptb_xl_data(
        db_dir=db_dir,
        fs='hr',
        category=category,
        remove_empty=True,
        folds=folds,
        verbose=verbose)
    if normalize:
        normalize = functools.partial(
            physionet.normalize_ptb_xl, inplace=True)
    data_set = _prepare_data(
        records,
        labels,
        normalize_fn=normalize,
        fs=fs,
        pad=pad,
        verbose=verbose)
    return data_set


def _prepare_data(records, labels, normalize_fn=None, fs=None, pad=None, verbose=False):
    x = _transform_records(
        records,
        fs=fs,
        pad=pad,
        normalize=normalize_fn,
        verbose=verbose)
    data_set = {'x': x,
                'y': labels.to_numpy(),
                'record_ids': labels.index.to_numpy(),
                'classes': labels.columns.to_numpy()}
    return data_set


def _transform_records(records, fs=None, pad=None, normalize=None, verbose=False):
    if not normalize:
        def normalize(signal): return signal
    x = [normalize(record.p_signal) for record in records]
    if fs:
        x = physionet.resample_records(records, fs=fs, verbose=verbose)
    if pad:
        max_len = None if pad == 'max' else pad
        x = pad_sequences(x, max_len=max_len, padding='pre')
    x = np.array(x)
    return x
