import tensorflow as tf


def build_input_tensor_from_shape(shape, dtype=None, ignore_batch_dim=False):
    """ Build input tensor from shape which can be used to initialize the weights of a model. """
    if isinstance(shape, (list, tuple)):
        return [build_input_tensor_from_shape(shape=shape[i],
                                              dtype=dtype[i] if dtype else None,
                                              ignore_batch_dim=ignore_batch_dim)
                for i in range(len(shape))]
    elif isinstance(shape, dict):
        return {k: build_input_tensor_from_shape(shape=shape[k],
                                                 dtype=dtype[k] if dtype else None,
                                                 ignore_batch_dim=ignore_batch_dim)
                for k in shape}
    else:
        if ignore_batch_dim:
            shape = shape[1:]
        return tf.keras.layers.Input(shape, dtype=dtype)
