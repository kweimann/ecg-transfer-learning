import numpy as np
import sklearn.model_selection
import tensorflow as tf

from transplant.modules.resnet1d import ResNet, BottleneckBlock


def ecg_feature_extractor(arch=None, stages=None):
    if arch is None or arch == 'resnet18':
        resnet = ResNet(num_outputs=None,
                        blocks=(2, 2, 2, 2)[:stages],
                        kernel_size=(7, 5, 5, 3),
                        include_top=False)
    elif arch == 'resnet34':
        resnet = ResNet(num_outputs=None,
                        blocks=(3, 4, 6, 3)[:stages],
                        kernel_size=(7, 5, 5, 3),
                        include_top=False)
    elif arch == 'resnet50':
        resnet = ResNet(num_outputs=None,
                        blocks=(3, 4, 6, 3)[:stages],
                        kernel_size=(7, 5, 5, 3),
                        block_fn=BottleneckBlock,
                        include_top=False)
    else:
        raise ValueError('unknown architecture: {}'.format(arch))
    feature_extractor = tf.keras.Sequential([
        resnet,
        tf.keras.layers.GlobalAveragePooling1D()
    ])
    return feature_extractor


def train_test_split(data_set, **options):
    x, y, record_ids, classes = \
        data_set['x'], data_set['y'], data_set['record_ids'], data_set['classes']
    assert len(x) == len(y) == len(record_ids)
    idx = np.arange(len(x))
    train_idx, test_idx = sklearn.model_selection.train_test_split(idx, **options)
    train = {'x': x[train_idx],
             'y': y[train_idx],
             'record_ids': record_ids[train_idx],
             'classes': classes}
    test = {'x': x[test_idx],
            'y': y[test_idx],
            'record_ids': record_ids[test_idx],
            'classes': classes}
    return train, test
