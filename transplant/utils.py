import gzip
import pickle

import numpy as np
import pandas as pd


def pad_sequences(x, max_len=None, padding='pre'):
    """
    Pads sequences shorter than `max_len` and trims those longer than `max_len`.

    @param x: Array of sequences.
    @param max_len: Maximal length of the padded sequences. Defaults to the longest sequence.
    @param padding: Type of padding: 'pre' before sequence, 'post' after sequence.

    @return: Array of shape (num_sequences, max_len) containing the padded sequences.
    """
    if max_len is None:
        max_len = max(map(len, x))
    x_shape = x[0].shape
    x_dtype = x[0].dtype
    x_padded = np.zeros((len(x), max_len) + x_shape[1:], dtype=x_dtype)
    for i, x_i in enumerate(x):
        trim_len = min(max_len, len(x_i))
        if padding == 'pre':
            x_padded[i, -trim_len:] = x_i[-trim_len:]
        elif padding == 'post':
            x_padded[i, :trim_len] = x_i[:trim_len]
        else:
            raise ValueError('Unknown padding: %s' % padding)
    return x_padded


def create_predictions_frame(y_prob, y_true=None, y_pred=None, class_names=None, record_ids=None):
    """
    Create predictions matrix.

    @param y_prob: Float array with class probabilities of shape (num_samples,) or (num_samples, num_classes).
    @param y_true: Integer array with true labels of shape (num_samples,) or (num_samples, num_classes).
    @param y_pred: Integer array with class predictions of shape (num_samples,) or (num_samples, num_classes).
    @param class_names: Array of class names of shape (num_classes,).
    @param record_ids: Array of record names of shape (num_samples,).

    @return: DataFrame that contains the predictions matrix.
    """
    y_prob = np.squeeze(y_prob)
    if y_prob.ndim == 1:  # binary classification
        y_prob = np.stack([1 - y_prob, y_prob], axis=1)
    num_classes = y_prob.shape[1]
    if class_names is None:
        # use index of the label as a class name
        class_names = np.arange(num_classes)
    elif len(class_names) != num_classes:
        raise ValueError('length of class_names does not match with the number of classes')
    columns = ['prob_{}'.format(label) for label in class_names]
    data = {column: y_prob[:, i] for i, column in enumerate(columns)}
    if y_pred is not None:
        y_pred = np.squeeze(y_pred)
        if y_pred.ndim == 1:
            y_pred = np.stack([1 - y_pred, y_pred], axis=1)
        if y_pred.shape != y_prob.shape:
            raise ValueError('y_prob and y_pred shapes do not match')
        y_pred_columns = ['pred_{}'.format(label) for label in class_names]
        y_pred_data = {column: y_pred[:, i] for i, column in enumerate(y_pred_columns)}
        columns = columns + y_pred_columns
        data = {**data, **y_pred_data}
    if y_true is not None:
        y_true = np.squeeze(y_true)
        if y_true.ndim == 1:  # class indices
            # search for true labels that do not correspond to any column in the predictions matrix
            unknown_labels = np.setdiff1d(y_true, np.arange(num_classes))
            if len(unknown_labels) > 0:
                raise ValueError('Unknown labels encountered: %s' % unknown_labels)
            y_true = np.eye(num_classes)[y_true]
        if y_true.shape != y_prob.shape:
            raise ValueError('y_prob and y_true shapes do not match')
        y_true_columns = ['true_{}'.format(label) for label in class_names]
        y_true_data = {column: y_true[:, i] for i, column in enumerate(y_true_columns)}
        columns = y_true_columns + columns
        data = {**data, **y_true_data}
    predictions_frame = pd.DataFrame(
        data=data,
        columns=columns)
    if record_ids is not None:
        predictions_frame.insert(0, 'record_name', record_ids)
    return predictions_frame


def read_predictions(file):
    """
    Read predictions matrix.

    @param file: path to the csv file with predictions.

    @return: dictionary with keys: `y_prob`, (optionally) `y_true`, (optionally) `y_pred`, and `classes`.
    """
    df = pd.read_csv(file)
    classes = [label[5:] for label in df.columns if label.startswith('prob')]
    predictions = {}
    for prefix in ['true', 'pred', 'prob']:
        col_names = ['{}_{}'.format(prefix, label) for label in classes]
        col_names = [name for name in col_names if name in df.columns]
        if col_names:
            predictions['y_{}'.format(prefix)] = df[col_names].values
    predictions['classes'] = classes
    return predictions


def matches_spec(o, spec, ignore_batch_dim=False):
    """
    Test whether data object matches the desired spec.

    @param o: Data object.
    @param spec: Metadata for describing the the data object.
    @param ignore_batch_dim: Ignore first dimension when checking the shapes.

    @return: True if the data object matches the spec, otherwise False.
    """
    if isinstance(spec, (list, tuple)):
        if not isinstance(o, (list, tuple)):
            raise ValueError('data object is not a list or tuple which is required by the spec: {}'.format(spec))
        if len(spec) != len(o):
            raise ValueError('data object has a different number of elements than the spec: {}'.format(spec))
        for i in range(len(spec)):
            if not matches_spec(o[i], spec[i], ignore_batch_dim=ignore_batch_dim):
                return False
        return True
    elif isinstance(spec, dict):
        if not isinstance(o, dict):
            raise ValueError('data object is not a dict which is required by the spec: {}'.format(spec))
        if spec.keys() != o.keys():
            raise ValueError('data object has different keys than those specified in the spec: {}'.format(spec))
        for k in spec:
            if not matches_spec(o[k], spec[k], ignore_batch_dim=ignore_batch_dim):
                return False
            return True
    else:
        spec_shape = spec.shape[1:] if ignore_batch_dim else spec.shape
        o_shape = o.shape[1:] if ignore_batch_dim else o.shape
        return spec_shape == o_shape and spec.dtype == o.dtype


def running_mean_std(iterator, dtype=None):
    """
    Calculate mean and standard deviation while iterating over the data iterator.
    @param iterator: Data iterator.
    @param dtype: Type of accumulators.
    @return: Mean, Std.
    """
    sum_x = np.zeros((), dtype=dtype)
    sum_x2 = np.zeros((), dtype=dtype)
    n = 0
    for x in iterator:
        sum_x += np.sum(x, dtype=dtype)
        sum_x2 += np.sum(x ** 2, dtype=dtype)
        n += x.size
    mean = sum_x / n
    std = np.math.sqrt((sum_x2 / n) - (mean ** 2))
    return mean, std


def buffered_generator(generator, buffer_size):
    """
    Buffer the elements yielded by a generator. New elements replace the oldest elements in the buffer.
    The buffer should be accessed in a read-only manner.

    @param generator: Generator object.
    @param buffer_size: Number of elements in the buffer.

    @return: Generator that yields a buffer.
    """
    buffer = []
    for e in generator:
        buffer.append(e)
        if len(buffer) == buffer_size:
            break
    yield buffer
    for e in generator:
        buffer = buffer[1:] + [e]
        yield buffer


def load_pkl(file, compress=True):
    """ Load pickled file. """
    if compress:
        with gzip.open(file, 'rb') as fh:
            return pickle.load(fh)
    else:
        with open(file, 'rb') as fh:
            return pickle.load(fh)


def save_pkl(file, compress=True, **kwargs):
    """ Save dictionary in a pickle file. """
    if compress:
        with gzip.open(file, 'wb') as fh:
            pickle.dump(kwargs, fh, protocol=4)
    else:
        with open(file, 'wb') as fh:
            pickle.dump(kwargs, fh, protocol=4)


def is_multiclass(labels):
    """ Return true if this is a multiclass task otherwise false. """
    return labels.squeeze().ndim == 2 and any(labels.sum(axis=1) != 1)
