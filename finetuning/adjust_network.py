import argparse
from pathlib import Path

import tensorflow as tf

from finetuning.utils import ecg_feature_extractor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=Path, required=True,
                        help='Path to pretrained weights or a checkpoint of the model.')
    parser.add_argument('--channels', type=int, required=True, help='New number of input channels.')
    parser.add_argument('--arch', default='resnet18', help='Network architecture: '
                                                           '`resnet18`, `resnet34` or `resnet50`.')
    parser.add_argument('--stages', type=int, default=None, help='Stages of the residual network '
                                                                 'that will be pretrained.')
    args, _ = parser.parse_known_args()

    # initialize model
    model = ecg_feature_extractor(arch=args.arch, stages=args.stages)
    inputs = tf.keras.layers.Input((256, 1), dtype='float32')
    model(inputs)

    # load checkpoint
    model.load_weights(str(args.weights_file))

    # duplicate filters for each channel and scale them
    conv1_filters = model.weights[0]
    conv1_filters = tf.tile(conv1_filters, (1, args.channels, 1)) / args.channels

    # update the weights to support new input format
    weights = model.get_weights()
    weights[0] = conv1_filters

    # initialize updated model
    updated_model = ecg_feature_extractor(arch=args.arch, stages=args.stages)
    inputs = tf.keras.layers.Input((256, args.channels), dtype='float32')
    updated_model(inputs)

    # set weights
    updated_model.set_weights(weights)

    # save updated model
    updated_weights_file = args.weights_file.parent / (args.weights_file.name + '.adjusted')
    updated_model.save_weights(str(updated_weights_file))
