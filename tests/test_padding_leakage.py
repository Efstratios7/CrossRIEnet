import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf

from crossrie.custom_layers import Two_Stream_EncoderLayer


def test_two_stream_encoder_masks_padded_tail_from_valid_prefix():
    tf.keras.utils.set_random_seed(123)
    layer = Two_Stream_EncoderLayer(
        encoding_units=[],
        lstm_units=[4],
        final_hidden_layer_sizes=[],
        final_activation="linear",
    )

    valid_prefix = tf.constant(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],
        dtype=tf.float32,
    )
    zero_padding = tf.zeros((1, 3, 3), dtype=tf.float32)
    garbage_padding = tf.ones((1, 3, 3), dtype=tf.float32) * 10.0

    seq_with_zero_padding = tf.concat([valid_prefix, zero_padding], axis=1)
    seq_with_garbage_padding = tf.concat([valid_prefix, garbage_padding], axis=1)

    out_zero = layer(
        [seq_with_zero_padding, seq_with_zero_padding],
        training=False,
    )
    out_garbage = layer(
        [seq_with_garbage_padding, seq_with_garbage_padding],
        training=False,
    )

    np.testing.assert_allclose(
        out_zero[:, :2].numpy(),
        out_garbage[:, :2].numpy(),
        atol=1e-6,
    )
