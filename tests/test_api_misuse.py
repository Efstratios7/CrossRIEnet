import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pytest
import tensorflow as tf

from crossrie import CrossRIELayer
from crossrie.custom_layers import CustomNormalizationLayer


def test_outputs_must_not_be_empty():
    with pytest.raises(ValueError, match="outputs"):
        CrossRIELayer(outputs=[])


def test_outputs_rejects_unknown_keys_even_when_valid_key_present():
    with pytest.raises(ValueError, match="outputs"):
        CrossRIELayer(outputs=["BAD", "Cxy"])


def test_outputs_rejects_string_to_avoid_accidental_dict_return():
    with pytest.raises((TypeError, ValueError), match="outputs"):
        CrossRIELayer(outputs="Cxy")


def test_integer_inputs_fail_with_clear_float_dtype_error():
    layer = CrossRIELayer(
        encoding_units=[2],
        lstm_units=[2],
        final_hidden_layer_sizes=[],
    )
    Cxx = tf.eye(2, batch_shape=[1], dtype=tf.int32)
    Cyy = tf.eye(2, batch_shape=[1], dtype=tf.int32)
    Cxy = tf.eye(2, batch_shape=[1], dtype=tf.int32)
    T_samples = tf.constant([10], dtype=tf.int32)

    with pytest.raises((TypeError, ValueError), match="float"):
        layer([Cxx, Cyy, Cxy, T_samples])


def test_custom_normalization_preserves_float64_dtype():
    x = tf.constant([[[1.0], [2.0]]], dtype=tf.float64)
    y = CustomNormalizationLayer(mode="sum", name="norm")(x)

    assert y.dtype == tf.float64
