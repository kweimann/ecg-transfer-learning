"""
Implementation of the Transformer [1] based on: https://www.tensorflow.org/tutorials/text/transformer

[1] https://arxiv.org/abs/1706.03762
"""

import numpy as np
import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.W_v = tf.keras.layers.Dense(d_model)
        self.W_k = tf.keras.layers.Dense(d_model)
        self.W_q = tf.keras.layers.Dense(d_model)
        self.W_o = tf.keras.layers.Dense(d_model)

    def scaled_dot_product_attention(self, v, k, q, mask=None):
        dk = tf.cast(self.d_model, tf.float32)
        attn_logits = tf.matmul(q, k, transpose_b=True) / tf.sqrt(dk)
        if mask is not None:
            attn_logits += mask * -1e9
        attn_weights = tf.nn.softmax(attn_logits, axis=-1)
        scaled_attn = tf.matmul(attn_weights, v)
        return scaled_attn, attn_weights

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.shape
        x = tf.reshape(x, (-1, seq_len, self.num_heads, self.depth))
        x = tf.transpose(x, (0, 2, 1, 3))
        return x

    def concat_heads(self, x):
        batch_size, num_heads, seq_len, d_model = x.shape
        x = tf.transpose(x, (0, 2, 1, 3))
        x = tf.reshape(x, (-1, seq_len, self.d_model))
        return x

    def call(self, v, k, q, mask=None):
        v = self.split_heads(self.W_v(v))
        k = self.split_heads(self.W_k(k))
        q = self.split_heads(self.W_q(q))
        scaled_attn, attn_weights = self.scaled_dot_product_attention(v, k, q, mask)
        scaled_attn = self.concat_heads(scaled_attn)
        mh_attn = self.W_o(scaled_attn)
        return mh_attn, attn_weights


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.ff1 = tf.keras.layers.Dense(dff)
        self.relu1 = tf.keras.layers.ReLU()
        self.ff2 = tf.keras.layers.Dense(d_model)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.layernorm2 = tf.keras.layers.LayerNormalization()

    def call(self, x, training=None, mask=None):
        shortcut = x
        x, _ = self.mha(x, x, x, mask)
        x = self.dropout1(x, training=training)
        x = self.layernorm1(x + shortcut)
        shortcut = x
        x = self.ff1(x)
        x = self.relu1(x)
        x = self.ff2(x)
        x = self.dropout2(x, training=training)
        x = self.layernorm2(x + shortcut)
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers=6, d_model=512, num_heads=8, dff=2048,
                 max_seq_len=512, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.pos_embedding = positional_embedding(max_seq_len, d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout)
                           for _ in range(num_layers)]

    def call(self, x, training=None, mask=None):
        """
        @param x: (batch_size, seq_len, d_model)
        @param training: training mode
        @param mask: (batch_size, seq_len)
        @return: (batch_size, seq_len, d_model)
        """
        if mask is not None:
            # add extra dimensions to add the padding to the attention logits
            mask = mask[:, None, None, :]
        _, seq_len, d_model = x.shape
        dk = tf.cast(self.d_model, tf.float32)
        x *= tf.sqrt(dk)
        x += self.pos_embedding[None, :seq_len]  # add batch dimension
        x = self.dropout(x, training=training)
        for enc_layer in self.enc_layers:
            x = enc_layer(x, training=training, mask=mask)
        return x


def positional_embedding(size, d_model):
    indices = np.arange(size)[:, None]
    i = np.arange(d_model)[None, :]
    angle_rads = indices / np.power(10000, (2 * (i // 2)) / d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_embedding = tf.cast(angle_rads, tf.float32)
    return pos_embedding
