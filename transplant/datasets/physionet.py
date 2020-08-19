import ast
from pathlib import Path

import numpy as np
import pandas as pd
import samplerate
import wfdb
import wfdb.processing
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

challenge17_mean = 0.0075
challenge17_std = 0.2373

challenge20_mean = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
challenge20_std = [0.2137, 0.2356, 0.2205, 0.2086, 0.1962, 0.2139, 0.3359, 0.4167, 0.4489, 0.4828, 0.5230, 0.5614]

ptb_xl_mean = [-0.0018, -0.0013, 0.0005, 0.0015, -0.0011, -0.0004, 0.0002, -0.0009, -0.0015, -0.0018, -0.0012, -0.0012]
ptb_xl_std = [0.1921, 0.1677, 0.1791, 0.1454, 0.1553, 0.1500, 0.2339, 0.3384, 0.3351, 0.3101, 0.2901, 0.2464]


def read_challenge17_data(db_dir, verbose=False):
    """
    Read the PhysioNet Challenge 2017 data set.

    @param db_dir: Database directory.
    @param verbose: Whether to show a progress bar.

    @return: Tuple of: (array of wfdb records, DataFrame with record ids as index and one hot encoded labels as data).
    """
    db_dir = Path(db_dir)
    if not db_dir.is_dir():
        raise ValueError('Provided path is not a directory: %s' % db_dir)
    index_file = db_dir / 'RECORDS'
    reference_file = db_dir / 'REFERENCE.csv'
    if not index_file.is_file():
        raise ValueError('Index file does not exist')
    if not reference_file.is_file():
        raise ValueError('Reference file does not exist')
    records_index = pd.read_csv(index_file, names=['record_name'], dtype='str', index_col='record_name')
    references = pd.read_csv(reference_file, names=['record_name', 'label'], index_col='record_name', dtype='str')
    references = pd.merge(records_index, references, on='record_name')
    label_df = pd.get_dummies(references.label)
    records_iterator = references.iterrows()
    if verbose:
        records_iterator = tqdm(records_iterator, total=len(references), desc='Reading records')
    records = []
    for record_name, _ in records_iterator:
        record_file = db_dir / record_name
        record = wfdb.rdrecord(str(record_file))
        records.append(record)
    return records, label_df


def read_challenge20_data(db_dir, verbose=False):
    """
    Read the PhysioNet Challenge 2020 data set.

    @param db_dir: Database directory.
    @param verbose: Whether to show a progress bar.

    @return: Tuple of: (array of wfdb records, DataFrame with record ids as index and one hot encoded labels as data).
    """
    db_dir = Path(db_dir)
    if not db_dir.is_dir():
        raise ValueError('Provided path is not a directory: %s' % db_dir)
    record_names = [file.stem for file in db_dir.iterdir()
                    if file.suffix == '.mat']
    if verbose:
        record_names = tqdm(record_names, desc='Reading records')
    records = []
    labels = []
    for record_name in record_names:
        record = wfdb.rdrecord(str(db_dir / record_name))
        records.append(record)
        for comment in record.comments:
            parts = comment.partition('Dx:')  # (before, string, after)
            if parts[1]:
                multi_label = [label.strip() for label in parts[2].split(',')]
                labels.append(multi_label)
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(labels)
    labels_df = pd.DataFrame(dict(zip(mlb.classes_, labels.T)), index=record_names)
    return records, labels_df


def read_ptb_xl_data(db_dir, fs='hr', category='rhythm', remove_empty=True, folds=None, verbose=False):
    """
    Read the PhysioNet PTB-XL data set.

    @param db_dir: Database directory.
    @param fs: Sampling frequency: 'lr' or 'hr' (default).
    @param category: Statement category.
    @param remove_empty: Whether to remove records with no statement in given category.
    @param folds: Only use data from the selected folds.
    @param verbose: Whether to show a progress bar.

    @return: Tuple of: (array of wfdb records, DataFrame with record ids as index and one hot encoded labels as data).
    """
    def get_labels(scp_codes_str):
        scp_codes = ast.literal_eval(scp_codes_str)
        return [label in scp_codes for label in labels]
    if fs not in ['lr', 'hr']:
        raise ValueError('Available sampling frequencies are: \'lr\' and \'hr\'.')
    db_dir = Path(db_dir)
    if not db_dir.is_dir():
        raise ValueError('Provided path is not a directory: %s' % db_dir)
    reference_file = db_dir / 'ptbxl_database.csv'
    statement_file = db_dir / 'scp_statements.csv'
    if not reference_file.is_file():
        raise ValueError('Reference file does not exist')
    if not statement_file.is_file():
        raise ValueError('Statement file does not exist')
    references = pd.read_csv(reference_file, index_col='ecg_id')
    if folds:
        references = references[references.strat_fold.isin(folds)]
    stmt = pd.read_csv(statement_file, index_col=0)
    labels = stmt[stmt[category] == 1].index.to_numpy()
    label_matrix = np.array(references.scp_codes.apply(get_labels).tolist(), dtype='int32')
    label_df = pd.DataFrame(
        data=label_matrix,
        index=references.index,
        columns=labels)
    if remove_empty:
        non_empty = label_df.sum(axis=1) > 0
        label_df = label_df[non_empty]
    filenames = references.loc[label_df.index][['filename_' + fs]]
    if verbose:
        filenames = tqdm(filenames.iterrows(), desc='Reading records', total=len(filenames))
    else:
        filenames = filenames.iterrows()
    records = []
    for ecg_id, row in filenames:
        record_file = db_dir / row['filename_' + fs]
        record = wfdb.rdrecord(str(record_file))
        record.record_name = ecg_id
        records.append(record)
    return records, label_df


def resample_records(records, fs, verbose=False):
    """
    Resample records at the desired sampling rate.

    @param records: Array of wfdb recordings.
    @param fs: The desired sampling rate.
    @param verbose: Whether to show a progress bar.

    @return: Array of resampled signals.
    """
    if verbose:
        records = tqdm(records, desc='Resampling records')
    signals = []
    for record in records:
        signal = record.p_signal
        if fs != record.fs:
            fs_ratio = fs / record.fs
            signal = samplerate.resample(signal, fs_ratio)
        signals.append(signal)
    return signals


def normalize_challenge17(array, inplace=False):
    """
    Normalize an array using the mean and standard deviation calculated over
     the entire PhysioNet Challenge 2017 dataset.

    @param array: Numpy array to normalize.
    @param inplace: Whether to perform the normalization steps in-place.

    @return: Normalized array.
    """
    return _normalize(array, challenge17_mean, challenge17_std, inplace=inplace)


def normalize_challenge20(array, inplace=False):
    """
    Normalize an array using the mean and standard deviation calculated over
     the entire PhysioNet Challenge 2020 dataset.

    @param array: Numpy array to normalize.
    @param inplace: Whether to perform the normalization steps in-place.

    @return: Normalized array.
    """
    return _normalize(array, challenge20_mean, challenge20_std, inplace=inplace)


def normalize_ptb_xl(array, inplace=False):
    """
    Normalize an array using the mean and standard deviation calculated over
     the entire PhysioNet PTB-XL dataset.

    @param array: Numpy array to normalize.
    @param inplace: Whether to perform the normalization steps in-place.

    @return: Normalized array.
    """
    return _normalize(array, ptb_xl_mean, ptb_xl_std, inplace=inplace)


def _normalize(array, mean, std, inplace=False):
    if inplace:
        array -= mean
        array /= std
    else:
        array = (array - mean) / std
    return array
