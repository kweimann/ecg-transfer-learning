import functools
import multiprocessing
import os

import tensorflow as tf
from tqdm import tqdm

from finetuning.utils import ecg_feature_extractor
from transplant.datasets import icentia11k
from transplant.modules.resnet1d import ResidualBlock, BottleneckBlock
from transplant.modules.transformer import Encoder
from transplant.modules.utils import build_input_tensor_from_shape
from transplant.tasks.cpc import CPCSolver


def unzip_icentia11k(db_dir, patient_ids, out_dir, num_workers=1, patients_per_worker=1, verbose=False):
    os.makedirs(out_dir, exist_ok=True)
    unzip_patient_fn = functools.partial(icentia11k.unzip_patient_data, db_dir, out_dir=out_dir)
    if num_workers == 1:
        unzipped_patients = map(unzip_patient_fn, patient_ids)
    else:
        pool = multiprocessing.Pool(num_workers)
        unzipped_patients = pool.imap_unordered(unzip_patient_fn, patient_ids, chunksize=patients_per_worker)
    if verbose:
        unzipped_patients = tqdm(unzipped_patients, desc='Unzipping patients', total=len(patient_ids))
    for _ in unzipped_patients:
        pass


def task_solver(task, arch='resnet18', stages=None, return_feature_extractor=False):
    feature_extractor = ecg_feature_extractor(arch, stages=stages)
    last_residual_block = feature_extractor.layers[0].layers[-1]
    if isinstance(last_residual_block, ResidualBlock):
        d_model = last_residual_block.filters
    elif isinstance(last_residual_block, BottleneckBlock):
        d_model = last_residual_block.filters * last_residual_block.expansion
    else:
        raise ValueError('Feature extractor is not a residual network')
    if task == 'rhythm':
        num_classes = len(icentia11k.ds_rhythm_names)
        model = tf.keras.Sequential([
            feature_extractor,
            tf.keras.layers.Dense(num_classes),
        ])
    elif task == 'beat':
        num_classes = len(icentia11k.ds_beat_names)
        model = tf.keras.Sequential([
            feature_extractor,
            tf.keras.layers.Dense(num_classes)
        ])
    elif task == 'hr':
        num_classes = len(icentia11k.ds_hr_names)
        model = tf.keras.Sequential([
            feature_extractor,
            tf.keras.layers.Dense(num_classes)
        ])
    elif task == 'cpc':
        model = CPCSolver(signal_embedding=feature_extractor,
                          transformer=Encoder(num_layers=3, d_model=d_model, num_heads=8,
                                              dff=2 * d_model, dropout=0.))
    else:
        raise ValueError('unknown task: {}'.format(task))
    if return_feature_extractor:
        return model, feature_extractor
    else:
        return model


def get_pretrained_weights(checkpoint_file, task, arch='resnet18', stages=None):
    model, feature_extractor = task_solver(task, arch, stages=stages, return_feature_extractor=True)
    if task in ['rhythm', 'beat', 'hr']:
        inputs = build_input_tensor_from_shape(tf.TensorShape((None, 1)))
    elif task == 'cpc':
        # exact shapes do not matter during the initialization
        inputs = build_input_tensor_from_shape({'context': tf.TensorShape((1, 1, 1)),
                                                'samples': tf.TensorShape((1, 1, 1))})
    else:
        raise ValueError('unknown task: {}'.format(task))
    model(inputs)
    model.load_weights(checkpoint_file)
    return feature_extractor
