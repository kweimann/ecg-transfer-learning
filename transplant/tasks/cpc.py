import tensorflow as tf

from transplant.modules.attn_pool import AttentionPooling
from transplant.tasks.utils import embed_frames


class CPCSolver(tf.keras.Model):
    def __init__(self, signal_embedding, transformer, **kwargs):
        """
        @param signal_embedding: (batch_size, *frame_shape) -> (batch_size, d_model)
        @param transformer: (batch_size, num_frames, d_model) -> (batch_size, num_frames, d_model)
        """
        super().__init__(**kwargs)
        self.signal_embedding = signal_embedding
        self.attn_pooling = AttentionPooling(transformer, keepdims=True)

    def call(self, x, training=None, **kwargs):
        """
        @param x:
            Input dictionary:
                context: frames in the context
                samples: sample frames to predict the future from

            Input shapes:
                context: (batch_size, context_size, *frame_shape)
                samples: (batch_size, num_samples, *frame_shape)
        @param training: training mode
        @return: Score of each sample which describes how well it matches the future.

            Output shape: (batch_size, num_samples)
        """
        context, samples = x['context'], x['samples']
        # extract features from context frames using the signal embedding model
        context = embed_frames(context, self.signal_embedding)  # (batch_size, context_size, d_model)
        # pool embedded context frames into a single context vector
        context = self.attn_pooling(context, training=training)  # (batch_size, 1, d_model)
        # extract features from samples using the signal embedding model
        samples = embed_frames(samples, self.signal_embedding)  # (batch_size, num_samples, d_model)
        # score how likely a sample is a positive sample
        samples = tf.transpose(samples, (0, 2, 1))  # (batch_size, d_model, num_samples)
        scores = tf.matmul(context, samples)
        scores = tf.squeeze(scores, axis=1)
        return scores
