import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from finetuning.utils import ecg_feature_extractor, train_test_split
from transplant.evaluation import auc, f1, multi_f1, CustomCheckpoint
from transplant.utils import (
    create_predictions_frame,
    load_pkl,
    is_multiclass
)


def _create_dataset_from_data(data):
    return tf.data.Dataset.from_tensor_slices((data['x'], data['y']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-dir', type=Path, required=True, help='Job output directory.')
    parser.add_argument('--train', type=Path, required=True, help='Path to the train file.')
    parser.add_argument('--val', type=Path, help='Path to the validation file.\n'
                                                 'Overrides --val-size.')
    parser.add_argument('--test', type=Path, help='Path to the test file.')
    parser.add_argument('--weights-file', type=Path, help='Path to pretrained weights or a checkpoint of the model.')
    parser.add_argument('--val-size', type=float, default=None,
                        help='Size of the validation set or proportion of the train set.')
    parser.add_argument('--arch', default='resnet18', help='Network architecture: '
                                                           '`resnet18`, `resnet34` or `resnet50`.')
    parser.add_argument('--subset', type=float, default=None, help='Size of a subset of the train set '
                                                                   'or proportion of the train set.')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size.')
    parser.add_argument('--val-metric', default='loss',
                        help='Validation metric used to find the best model at each epoch. Supported metrics are:'
                             '`loss`, `acc`, `f1`, `auc`.')
    parser.add_argument('--channel', type=int, default=None, help='Use only the selected channel. '
                                                                  'By default use all available channels.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs.')
    parser.add_argument('--seed', type=int, default=None, help='Random state.')
    parser.add_argument('--verbose', action='store_true', help='Show debug messages.')
    args, _ = parser.parse_known_args()

    if args.val_metric not in ['loss', 'acc', 'f1', 'auc']:
        raise ValueError('Unknown metric: {}'.format(args.val_metric))

    os.makedirs(str(args.job_dir), exist_ok=True)
    print('Creating working directory in {}'.format(args.job_dir))

    seed = args.seed or np.random.randint(2 ** 16)
    print('Setting random state {}'.format(seed))
    np.random.seed(seed)

    if not args.val and args.val_size:
        if args.val_size >= 1:
            args.val_size = int(args.val_size)

    if args.subset and args.subset >= 1:
        args.subset = int(args.subset)

    print('Loading train data from {} ...'.format(args.train))
    train = load_pkl(str(args.train))

    if args.val:
        print('Loading validation data from {} ...'.format(args.val))
        val = load_pkl(str(args.val))
    elif args.val_size:
        original_train_size = len(train['x'])
        train, val = train_test_split(train, test_size=args.val_size, stratify=train['y'])
        new_train_size = len(train['x'])
        new_val_size = len(val['x'])
        print('Split data into train {:.2%} and validation {:.2%}'.format(
            new_train_size / original_train_size, new_val_size / original_train_size))
    else:
        val = None

    if args.test:
        print('Loading test data from {} ...'.format(args.test))
        test = load_pkl(str(args.test))
    else:
        test = None

    if args.subset:
        original_train_size = len(train['x'])
        train, _ = train_test_split(train, train_size=args.subset, stratify=train['y'])
        new_train_size = len(train['x'])
        print('Using only {:.2%} of train data'.format(new_train_size / original_train_size))

    if args.channel is not None:
        train['x'] = train['x'][:, :, args.channel:args.channel + 1]
        if val:
            val['x'] = val['x'][:, :, args.channel:args.channel + 1]
        if test:
            test['x'] = test['x'][:, :, args.channel:args.channel + 1]

    print('Train data shape:', train['x'].shape)

    train_data = _create_dataset_from_data(train).shuffle(len(train['x'])).batch(args.batch_size)
    val_data = _create_dataset_from_data(val).batch(args.batch_size) if val else None
    test_data = _create_dataset_from_data(test).batch(args.batch_size) if test else None

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        print('Building model ...')
        num_classes = len(train['classes'])

        if is_multiclass(train['y']):
            activation = 'sigmoid'
            loss = tf.keras.losses.BinaryCrossentropy()
            accuracy = tf.keras.metrics.BinaryAccuracy(name='acc')
        else:
            activation = 'softmax'
            loss = tf.keras.losses.CategoricalCrossentropy()
            accuracy = tf.keras.metrics.CategoricalAccuracy(name='acc')

        # add classification layer on top of the ecg feature extractor
        model = ecg_feature_extractor(args.arch)
        model.add(tf.keras.layers.Dense(num_classes, activation=activation))

        # initialize the weights of the model
        inputs = tf.keras.layers.Input(train['x'].shape[1:], dtype=train['x'].dtype)
        model(inputs)

        print('# model parameters: {:,d}'.format(model.count_params()))

        if args.weights_file:
            # initialize weights (excluding the optimizer state) to load the pretrained resnet
            # the optimizer state is randomly initialized in the `model.compile` function
            print('Loading weights from file {} ...'.format(args.weights_file))
            model.load_weights(str(args.weights_file))

        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=loss,
                      metrics=[accuracy])

        callbacks = []

        logger = tf.keras.callbacks.CSVLogger(str(args.job_dir / 'history.csv'))
        callbacks.append(logger)

        if args.val_metric in ['loss', 'acc']:
            monitor = ('val_' + args.val_metric) if val else args.val_metric
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=str(args.job_dir / 'best_model.weights'),
                monitor=monitor,
                save_best_only=True,
                save_weights_only=True,
                mode='auto',
                verbose=1)
        elif args.val_metric == 'f1':
            if is_multiclass(train['y']):
                score_fn = multi_f1
            else:
                score_fn = f1
            checkpoint = CustomCheckpoint(
                filepath=str(args.job_dir / 'best_model.weights'),
                data=(val_data, val['y']) if val else (train_data, train['y']),
                score_fn=score_fn,
                save_best_only=True,
                verbose=1)
        elif args.val_metric == 'auc':
            checkpoint = CustomCheckpoint(
                filepath=str(args.job_dir / 'best_model.weights'),
                data=(val_data, val['y']) if val else (train_data, train['y']),
                score_fn=auc,
                save_best_only=True,
                verbose=1)
        else:
            raise ValueError('Unknown metric: {}'.format(args.val_metric))

        callbacks.append(checkpoint)

        if val:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=50, verbose=1)
            callbacks.append(early_stopping)

        model.fit(train_data, epochs=args.epochs, verbose=2, validation_data=val_data, callbacks=callbacks)

        # load best model for inference
        print('Loading the best weights from file {} ...'.format(str(args.job_dir / 'best_model.weights')))
        model.load_weights(str(args.job_dir / 'best_model.weights'))

        print('Predicting training data ...')
        train_y_prob = model.predict(train['x'], batch_size=args.batch_size)
        train_predictions = create_predictions_frame(
            y_prob=train_y_prob,
            y_true=train['y'],
            class_names=train['classes'],
            record_ids=train['record_ids'])
        train_predictions.to_csv(args.job_dir / 'train_predictions.csv', index=False)

        if val:
            print('Predicting validation data ...')
            val_y_prob = model.predict(val['x'], batch_size=args.batch_size)
            val_predictions = create_predictions_frame(
                y_prob=val_y_prob,
                y_true=val['y'],
                class_names=train['classes'],
                record_ids=val['record_ids'])
            val_predictions.to_csv(args.job_dir / 'val_predictions.csv', index=False)

        if test:
            print('Predicting test data ...')
            test_y_prob = model.predict(test['x'], batch_size=args.batch_size)
            test_predictions = create_predictions_frame(
                y_prob=test_y_prob,
                y_true=test['y'],
                class_names=train['classes'],
                record_ids=test['record_ids'])
            test_predictions.to_csv(args.job_dir / 'test_predictions.csv', index=False)
