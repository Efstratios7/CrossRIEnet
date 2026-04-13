import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf

from crossrie import CrossRIELayer
from crossrie.custom_layers import reconstruct_matrix_from_svd, svd_via_eigh_full


def test_svd_zero_matrix_returns_orthonormal_right_basis():
    C = tf.zeros((1, 3, 5), dtype=tf.float64)

    s, U, V = svd_via_eigh_full(C)
    C_hat = reconstruct_matrix_from_svd(s, U, V)

    np.testing.assert_allclose(C_hat.numpy(), C.numpy(), atol=1e-12)
    np.testing.assert_allclose(
        tf.matmul(U, U, transpose_a=True).numpy(),
        np.eye(3)[None, :, :],
        atol=1e-10,
    )
    np.testing.assert_allclose(
        tf.matmul(V, V, transpose_a=True).numpy(),
        np.eye(5)[None, :, :],
        atol=1e-10,
    )


def test_svd_zero_matrix_gradient_is_finite():
    C = tf.zeros((1, 3, 5), dtype=tf.float64)

    with tf.GradientTape() as tape:
        tape.watch(C)
        s, _, _ = svd_via_eigh_full(C)
        loss = tf.reduce_sum(s)

    grad = tape.gradient(loss, C)

    assert grad is not None
    assert np.all(np.isfinite(grad.numpy()))


def test_svd_rank_deficient_gradient_is_finite():
    C = tf.ones((1, 4, 4), dtype=tf.float64)

    with tf.GradientTape() as tape:
        tape.watch(C)
        s, _, _ = svd_via_eigh_full(C)
        loss = tf.reduce_sum(s)

    grad = tape.gradient(loss, C)

    assert grad is not None
    assert np.all(np.isfinite(grad.numpy()))


def test_svd_huge_scale_inputs_remain_finite():
    C = tf.constant([[[1e154, 0.0], [0.0, 1.0]]], dtype=tf.float64)

    s, U, V = svd_via_eigh_full(C)
    C_hat = reconstruct_matrix_from_svd(s, U, V)

    assert np.all(np.isfinite(s.numpy()))
    assert np.all(np.isfinite(U.numpy()))
    assert np.all(np.isfinite(V.numpy()))
    assert np.all(np.isfinite(C_hat.numpy()))


def test_crossrie_extreme_asymmetric_dynamic_shapes_are_finite():
    tf.keras.utils.set_random_seed(5)
    layer = CrossRIELayer(
        encoding_units=[3],
        lstm_units=[3],
        final_hidden_layer_sizes=[2],
        multiplicative=False,
        final_activation="linear",
    )

    for N, M in [(1, 1), (1, 50), (50, 1), (2, 100), (100, 2)]:
        Cxx = tf.eye(N, batch_shape=[1], dtype=tf.float32)
        Cyy = tf.eye(M, batch_shape=[1], dtype=tf.float32)
        Cxy = tf.random.normal((1, N, M), dtype=tf.float32)
        T_samples = tf.constant([100.0], dtype=tf.float32)

        out = layer([Cxx, Cyy, Cxy, T_samples], training=False)

        assert out.shape == (1, N, M)
        assert np.all(np.isfinite(out.numpy()))
