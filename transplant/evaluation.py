import warnings

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, f1_score


def auc(y_true, y_prob):
    if y_prob.ndim != 2:
        raise ValueError('y_prob must be a 2d matrix with class probabilities for each sample')
    if y_true.shape != y_prob.shape:
        raise ValueError('shapes do not match')
    return roc_auc_score(y_true, y_prob, average='macro')


def f1(y_true, y_prob, multiclass=False, threshold=None):
    # threshold may also be a 1d array of thresholds for each class
    if y_prob.ndim != 2:
        raise ValueError('y_prob must be a 2d matrix with class probabilities for each sample')
    if y_true.ndim == 1:  # we assume that y_true is sparse (consequently, multiclass=False)
        if multiclass:
            raise ValueError('if y_true cannot be sparse and multiclass at the same time')
        depth = y_prob.shape[1]
        y_true = _one_hot(y_true, depth)
    if multiclass:
        if threshold is None:
            threshold = 0.5
        y_pred = y_prob >= threshold
    else:
        y_pred = y_prob >= np.max(y_prob, axis=1)[:, None]
    return f1_score(y_true, y_pred, average='macro')


def f_max(y_true, y_prob, thresholds=None):
    """ source: https://github.com/helme/ecg_ptbxl_benchmarking """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 100)
    pr, rc = macro_precision_recall(y_true, y_prob, thresholds)
    f1s = (2 * pr * rc) / (pr + rc)
    i = np.nanargmax(f1s)
    return f1s[i], thresholds[i]


def macro_precision_recall(y_true, y_prob, thresholds):  # multi-class multi-output
    """ source: https://github.com/helme/ecg_ptbxl_benchmarking """
    # expand analysis to the number of thresholds
    y_true = np.repeat(y_true[None, :, :], len(thresholds), axis=0)
    y_prob = np.repeat(y_prob[None, :, :], len(thresholds), axis=0)
    y_pred = y_prob >= thresholds[:, None, None]

    # compute true positives
    tp = np.sum(np.logical_and(y_true, y_pred), axis=2)

    # compute macro average precision handling all warnings
    with np.errstate(divide='ignore', invalid='ignore'):
        den = np.sum(y_pred, axis=2)
        precision = tp / den
        precision[den == 0] = np.nan
        with warnings.catch_warnings():  # for nan slices
            warnings.simplefilter("ignore", category=RuntimeWarning)
            av_precision = np.nanmean(precision, axis=1)

    # compute macro average recall
    recall = tp / np.sum(y_true, axis=2)
    av_recall = np.mean(recall, axis=1)

    return av_precision, av_recall


def challenge2020_metrics(y_true, y_pred, beta_f=2, beta_g=2, class_weights=None, single=False):
    """ source: https://github.com/helme/ecg_ptbxl_benchmarking """
    num_samples, num_classes = y_true.shape
    if single:  # if evaluating single class in case of threshold-optimization
        sample_weights = np.ones(num_samples)
    else:
        sample_weights = y_true.sum(axis=1)
    if class_weights is None:
        class_weights = np.ones(num_classes)
    f_beta = 0
    g_beta = 0
    for k, w_k in enumerate(class_weights):
        tp, fp, tn, fn = 0., 0., 0., 0.
        for i in range(num_samples):
            if y_true[i, k] == y_pred[i, k] == 1:
                tp += 1. / sample_weights[i]
            if y_pred[i, k] == 1 and y_true[i, k] != y_pred[i, k]:
                fp += 1. / sample_weights[i]
            if y_true[i, k] == y_pred[i, k] == 0:
                tn += 1. / sample_weights[i]
            if y_pred[i, k] == 0 and y_true[i, k] != y_pred[i, k]:
                fn += 1. / sample_weights[i]
        f_beta += w_k * ((1 + beta_f ** 2) * tp) / ((1 + beta_f ** 2) * tp + fp + beta_f ** 2 * fn)
        g_beta += w_k * tp / (tp + fp + beta_g * fn)
    f_beta /= class_weights.sum()
    g_beta /= class_weights.sum()
    return {'F_beta': f_beta,
            'G_beta': g_beta}


def _one_hot(x, depth):
    x_one_hot = np.zeros((x.size, depth))
    x_one_hot[np.arange(x.size), x] = 1
    return x_one_hot


def multi_f1(y_true, y_prob):
    return f1(y_true, y_prob, multiclass=True, threshold=0.5)


class CustomCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, data, score_fn, best=-np.Inf, save_best_only=False, batch_size=None, verbose=0):
        super().__init__()
        self.filepath = filepath
        self.data = data
        self.score_fn = score_fn
        self.save_best_only = save_best_only
        self.batch_size = batch_size
        self.verbose = verbose
        self.best = best

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        x, y_true = self.data
        y_prob = self.model.predict(x, batch_size=self.batch_size)
        score = self.score_fn(y_true, y_prob)
        logs.update({self.metric_name: score})
        filepath = self.filepath.format(epoch=epoch + 1, **logs)
        if score > self.best:
            if self.verbose:
                print('\nEpoch %05d: %s improved from %0.5f to %0.5f, saving model to %s'
                      % (epoch + 1, self.metric_name, self.best, score, filepath))
            self.model.save_weights(filepath, overwrite=True)
            self.best = score
        elif not self.save_best_only:
            if self.verbose:
                print('\nEpoch %05d: %s (%.05f) did not improve from %0.5f, saving model to %s'
                      % (epoch + 1, self.metric_name, score, self.best, filepath))
            self.model.save_weights(filepath, overwrite=True)
        else:
            if self.verbose:
                print('\nEpoch %05d: %s (%.05f) did not improve from %0.5f'
                      % (epoch + 1, self.metric_name, score, self.best))

    @property
    def metric_name(self):
        return self.score_fn.__name__
