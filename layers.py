from __future__ import print_function
import tensorflow as tf
import numpy as np


class MultiHeadAttention(tf.keras.layers.Layer):
    """ Attention Layer - multi-head scaled dot product attention (for encoder and decoder)

        Args:
            num_heads: number of attention heads which will be computed in parallel
            d_model: embedding size of output features

        Call arguments:
            q: query, shape (..., seq_len_q, depth_q)
            k: key, shape == (..., seq_len_k, depth_k)
            v: value, shape == (..., seq_len_v, depth_v)
            mask: Float tensor with shape broadcastable to (..., seq_len_q, seq_len_k) or None.

            Since we use scaled-product attention, we assume seq_len_k = seq_len_v

        Returns:
              attention outputs of shape (batch_size, seq_len_q, d_model)
    """

    def __init__(self, n_heads, d_model, **kwargs):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_depth = self.d_model // self.n_heads

        if self.d_model % self.n_heads != 0:
            raise ValueError("number of heads must divide d_model")

        # define weight matrices
        self.wq = tf.keras.layers.Dense(self.d_model, use_bias=False)  # (d_q, d_model)
        self.wk = tf.keras.layers.Dense(self.d_model, use_bias=False)  # (d_k, d_model)
        self.wv = tf.keras.layers.Dense(self.d_model, use_bias=False)  # (d_v, d_model)

        self.w_out = tf.keras.layers.Dense(self.d_model, use_bias=False)  # (d_model, d_model)

    def split_heads(self, tensor, batch_size):
        """Function for computing attention on several heads simultaneously
        Splits last dimension of a tensor into (num_heads, head_depth).
        Then we transpose it as (batch_size, num_heads, ..., head_depth) so that we can use broadcast
        """
        tensor = tf.reshape(tensor, (batch_size, -1, self.n_heads, self.head_depth))
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    # treats first parameter q as input, and  k, v as parameters, so input_shape=q.shape
    def call(self, q, k, v, mask=None):
        # shape of q: (batch_size, seq_len_q, d_q)
        batch_size = tf.shape(q)[0]

        # compute Q = q * w_q, ...
        Q = self.wq(q)  # (batch_size, seq_len_q, d_q) x (d_q, d_model) --> (batch_size, seq_len_q, d_model)
        K = self.wk(k)  # ... --> (batch_size, seq_len_k, d_model)
        V = self.wv(v)  # ... --> (batch_size, seq_len_v, d_model)

        # split heads: d_model = num_heads * head_depth + reshape
        Q = self.split_heads(Q, batch_size)  # (batch_size, num_heads, seq_len_q, head_depth)
        K = self.split_heads(K, batch_size)  # (batch_size, num_heads, seq_len_k, head_depth)
        V = self.split_heads(V, batch_size)  # (batch_size, num_heads, seq_len_v, head_depth)

        # similarity between context vector Q and key K // self-similarity in case of self-attention
        compatibility = tf.matmul(Q, K, transpose_b=True)  # (batch_size, num_heads, seq_len_q, seq_len_k)
                                                           # seq_len_q = n_nodes for encoder self-attention
                                                           # seq_len_q = 1 for decoder context-vector attention
                                                           # seq_len_k = n_nodes for both encoder & decoder
        # rescaling
        dk = tf.cast(tf.shape(K)[-1], tf.float32)
        compatibility = compatibility / tf.math.sqrt(dk)

        if mask is not None:
            # we need to reshape mask:
            # (batch_size, seq_len_q, seq_len_k) --> (batch_size, 1, seq_len_q, seq_len_k)
            # so that we will be able to do a broadcast:
            # (batch_size, num_heads, seq_len_q, seq_len_k) + (batch_size, 1, seq_len_q, seq_len_k)
            mask = mask[:, tf.newaxis, :, :]

            # we use tf.where since 0*-np.inf returns nan, but not -np.inf
            # compatibility = tf.where(
            #                     tf.broadcast_to(mask, compatibility.shape), tf.ones_like(compatibility) * (-np.inf),
            #                     compatibility
            #                      )

            compatibility = tf.where(mask,
                                    tf.ones_like(compatibility) * (-np.inf),
                                    compatibility)

        compatibility = tf.nn.softmax(compatibility, axis=-1)  # (batch_size, num_heads, seq_len_q, seq_len_k)

        # Replace NaN by zeros (tf.nn.softmax returns NaNs for masked rows)
        compatibility = tf.where(tf.math.is_nan(compatibility), tf.zeros_like(compatibility), compatibility)

        # seq_len_k = seq_len_v
        attention = tf.matmul(compatibility, V)  # (batch_size, num_heads, seq_len_q, head_depth)

        # transpose back to (batch_size, seq_len_q, num_heads, head_depth)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])

        # concatenate heads (last 2 dimensions)
        attention = tf.reshape(attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        # project output to the same dimension
        # this is equiv. to sum in the article (project heads with W_o and sum), beacuse of block-matrix multiplication
        #e.g. https://math.stackexchange.com/questions/2961550/matrix-block-multiplication-definition-properties-and-applications

        output = self.w_out(attention)  # (batch_size, seq_len_q, d_model)

        return output