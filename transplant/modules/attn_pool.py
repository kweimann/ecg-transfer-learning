import tensorflow as tf


class AttentionPooling(tf.keras.layers.Layer):
    def __init__(self, transformer, keepdims=False, **kwargs):
        """
        @param transformer: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        @param keepdims: If True then keep the output dimension of the transformer,
                otherwise squeeze the output.
        """
        super().__init__(**kwargs)
        self.transformer = transformer
        self.keepdims = keepdims
        initial_pool_token_embedding = tf.random.truncated_normal((transformer.d_model,))
        self.pool_token_embedding = tf.Variable(initial_pool_token_embedding, trainable=True)

    def call(self, x, training=None, mask=None):
        """
        @param x: (batch_size, seq_len, d_model)
        @param training: training mode
        @param mask: (batch_size, seq_len)
        @return: (batch_size, d_model) or (batch_size, 1, d_model) if keepdims is True
        """
        batch_size = tf.shape(x)[0]  # dynamic shape
        # prepend pool token embedding to each sample in the batch
        pool_token_embedding = tf.tile(self.pool_token_embedding, (batch_size,))
        pool_token_embedding = tf.reshape(pool_token_embedding, (batch_size, 1, self.transformer.d_model))
        x = tf.concat([pool_token_embedding, x], axis=1)  # x.shape == (batch_size, seq_len + 1, d_model)
        if mask is not None:
            # prepend a zero to each pad vector in the batch
            pool_token_embedding_mask = tf.zeros((batch_size, 1))
            mask = tf.concat([pool_token_embedding_mask, mask], axis=1)  # mask.shape == (batch_size, seq_len + 1)
        # encode the sequences using transformer
        x = self.transformer(x, training=training, mask=mask)
        # use pool token encoding as the pooled vector
        if self.keepdims:
            x = x[:, :1, :]
        else:
            x = x[:, 0, :]
        return x
