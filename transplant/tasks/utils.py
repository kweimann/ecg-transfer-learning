import tensorflow as tf


def embed_frames(x, embedding_layer):
    """
    @param x: (batch_size, num_frames, *frame_shape)
    @param embedding_layer: (batch_size, *frame_shape) -> (batch_size, d_model)
    @return: (batch_size, num_frames, d_model)
    """
    _, num_frames, *frame_shape = x.shape
    batch_size = tf.shape(x)[0]  # dynamic shape
    x = tf.reshape(x, (batch_size * num_frames, *frame_shape))
    x = embedding_layer(x)
    x = tf.reshape(x, (batch_size, num_frames, -1))
    return x
