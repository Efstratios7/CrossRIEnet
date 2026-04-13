"""
Unified test suite for CrossRIE.

Tests cover:
  - CrossRIELayer: forward/backward pass, architecture variants, activations,
    output shapes, dynamic dimensions, N/M > T regime, dataset pipeline.
  - svd_via_eigh_full: forward correctness, singular value properties,
    orthonormality, square/tall matrices.
  - Graph-mode gradients: gradient existence, varying dynamic shapes,
    gradient through reconstruction (verifies the tf.linalg.qr stop_gradient fix).
  - End-to-end layer gradients: train steps in additive/multiplicative modes,
    varying shapes, loss decrease.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import unittest
import numpy as np
import tensorflow as tf
from keras import losses, optimizers
from crossrie import CrossRIELayer
from crossrie.custom_layers import svd_via_eigh_full, reconstruct_matrix_from_svd


# ============================================================================
# CrossRIELayer unit tests
# ============================================================================

class TestCrossRIELayer(unittest.TestCase):
    """Fast unit tests for CrossRIELayer: verify plumbing, not learning."""

    def _build_and_train(self, layer, B=2, N=5, M=7, T=50):
        """Build a model, run one train step with tiny random data, and return the model + loss."""
        in_Cxx = tf.keras.Input(shape=(None, None))
        in_Cyy = tf.keras.Input(shape=(None, None))
        in_Cxy = tf.keras.Input(shape=(None, None))
        in_n = tf.keras.Input(shape=())

        out = layer([in_Cxx, in_Cyy, in_Cxy, in_n])
        model = tf.keras.Model(inputs=[in_Cxx, in_Cyy, in_Cxy, in_n], outputs=out)
        model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                       loss=losses.MeanSquaredError())

        Cxx = tf.random.normal((B, N, N), dtype=tf.float64)
        Cyy = tf.random.normal((B, M, M), dtype=tf.float64)
        Cxy = tf.random.normal((B, N, M), dtype=tf.float64)
        T_s = tf.constant([float(T)] * B, dtype=tf.float64)
        target = tf.random.normal((B, N, M), dtype=tf.float64)

        loss = model.train_on_batch([Cxx, Cyy, Cxy, T_s], target)
        return model, loss

    def test_model_initialization_additive(self):
        """Forward + backward pass in additive mode."""
        layer = CrossRIELayer(
            encoding_units=[8], lstm_units=[16], final_hidden_layer_sizes=[8],
            multiplicative=False, final_activation='linear'
        )
        _, loss = self._build_and_train(layer)
        self.assertGreater(loss, 0.0)

    def test_model_initialization_multiplicative(self):
        """Forward + backward pass in multiplicative mode."""
        layer = CrossRIELayer(
            encoding_units=[8], lstm_units=[16], final_hidden_layer_sizes=[8],
            multiplicative=True, final_activation='softplus'
        )
        _, loss = self._build_and_train(layer)
        self.assertGreater(loss, 0.0)

    def test_varying_architectures(self):
        """Different network depths and widths all complete a train step."""
        configs = [
            {'encoding': [32, 16], 'lstm': [64], 'final': [32]},
            {'encoding': [], 'lstm': [16], 'final': []},
            {'encoding': [8], 'lstm': [8, 8], 'final': [4]}
        ]
        for conf in configs:
            with self.subTest(config=conf):
                layer = CrossRIELayer(
                    encoding_units=conf['encoding'], lstm_units=conf['lstm'],
                    final_hidden_layer_sizes=conf['final'],
                    multiplicative=True, final_activation='softplus'
                )
                _, loss = self._build_and_train(layer)
                self.assertGreater(loss, 0.0)

    def test_valid_activations(self):
        """All supported multiplicative activations complete a train step."""
        for act in ['softplus', 'relu', 'sigmoid']:
            with self.subTest(activation=act):
                layer = CrossRIELayer(
                    encoding_units=[8], lstm_units=[16], final_hidden_layer_sizes=[8],
                    multiplicative=True, final_activation=act
                )
                _, loss = self._build_and_train(layer)
                self.assertGreater(loss, 0.0)

    def test_additive_activations(self):
        """All supported additive activations complete a train step."""
        for act in ['linear', 'tanh']:
            with self.subTest(activation=act):
                layer = CrossRIELayer(
                    encoding_units=[8], lstm_units=[16], final_hidden_layer_sizes=[8],
                    multiplicative=False, final_activation=act
                )
                _, loss = self._build_and_train(layer)
                self.assertGreater(loss, 0.0)

    def test_invalid_activation_config(self):
        """Validation rejects invalid activation/mode combos."""
        with self.assertRaises(ValueError):
            CrossRIELayer(multiplicative=True, final_activation='linear')

    def test_output_shape(self):
        """Output shape matches (B, N, M) for Cxy output."""
        B, N, M = 2, 6, 9
        layer = CrossRIELayer(
            encoding_units=[8], lstm_units=[16], final_hidden_layer_sizes=[8],
            multiplicative=False, final_activation='linear'
        )
        model, _ = self._build_and_train(layer, B=B, N=N, M=M)

        Cxx = tf.random.normal((B, N, N), dtype=tf.float64)
        Cyy = tf.random.normal((B, M, M), dtype=tf.float64)
        Cxy = tf.random.normal((B, N, M), dtype=tf.float64)
        T_s = tf.constant([100.0] * B, dtype=tf.float64)
        pred = model.predict([Cxx, Cyy, Cxy, T_s], verbose=0)
        self.assertEqual(pred.shape, (B, N, M))

    def test_varying_shapes(self):
        """Model handles different N, M across sequential train steps."""
        layer = CrossRIELayer(
            encoding_units=[8], lstm_units=[16], final_hidden_layer_sizes=[8],
            multiplicative=True, final_activation='softplus'
        )
        model, _ = self._build_and_train(layer, N=5, M=7)

        for N, M in [(8, 4), (3, 10), (6, 6)]:
            Cxx = tf.random.normal((2, N, N), dtype=tf.float64)
            Cyy = tf.random.normal((2, M, M), dtype=tf.float64)
            Cxy = tf.random.normal((2, N, M), dtype=tf.float64)
            T_s = tf.constant([100.0, 100.0], dtype=tf.float64)
            target = tf.random.normal((2, N, M), dtype=tf.float64)
            loss = model.train_on_batch([Cxx, Cyy, Cxy, T_s], target)
            self.assertGreater(loss, 0.0, f"Zero loss for N={N}, M={M}")

    def test_training_with_n_m_greater_than_t(self):
        """Model trains when both matrix dimensions exceed the sample count T."""
        B, N, M, T = 2, 12, 14, 5
        layer = CrossRIELayer(
            encoding_units=[8], lstm_units=[16], final_hidden_layer_sizes=[8],
            multiplicative=True, final_activation='softplus'
        )
        model, loss = self._build_and_train(layer, B=B, N=N, M=M, T=T)
        self.assertGreater(loss, 0.0)

        Cxx = tf.random.normal((B, N, N), dtype=tf.float64)
        Cyy = tf.random.normal((B, M, M), dtype=tf.float64)
        Cxy = tf.random.normal((B, N, M), dtype=tf.float64)
        T_s = tf.constant([float(T)] * B, dtype=tf.float64)
        target = tf.random.normal((B, N, M), dtype=tf.float64)
        loss2 = model.train_on_batch([Cxx, Cyy, Cxy, T_s], target)
        self.assertGreater(loss2, 0.0)

        pred = model.predict([Cxx, Cyy, Cxy, T_s], verbose=0)
        self.assertEqual(pred.shape, (B, N, M))

    def test_dataset_pipeline_integration(self):
        """One epoch / 1 step through the data generator pipeline still works."""
        from tests.data_generator import get_dynamic_dataset

        layer = CrossRIELayer(
            encoding_units=[8], lstm_units=[16], final_hidden_layer_sizes=[8],
            multiplicative=True, final_activation='softplus'
        )
        in_Cxx = tf.keras.Input(shape=(None, None))
        in_Cyy = tf.keras.Input(shape=(None, None))
        in_Cxy = tf.keras.Input(shape=(None, None))
        in_n = tf.keras.Input(shape=())
        out = layer([in_Cxx, in_Cyy, in_Cxy, in_n])
        model = tf.keras.Model(inputs=[in_Cxx, in_Cyy, in_Cxy, in_n], outputs=out)
        model.compile(optimizer='adam', loss='mse')

        dataset = get_dynamic_dataset(batch_size=2, N_range=(5, 8), M_range=(5, 8), ndays_range=(20, 30))
        history = model.fit(dataset, epochs=1, steps_per_epoch=1, verbose=0)
        self.assertEqual(len(history.history['loss']), 1)
        self.assertGreater(history.history['loss'][0], 0.0)


# ============================================================================
# SVD via eigh: forward correctness tests
# ============================================================================

class TestSvdViaEighFull(unittest.TestCase):
    """Tests for the svd_via_eigh_full function."""

    def test_forward_correctness(self):
        """SVD reconstruction should approximate the original matrix."""
        tf.random.set_seed(42)
        C = tf.random.normal((4, 6, 8), dtype=tf.float64)
        s_k, U_full, V_full = svd_via_eigh_full(C)
        C_hat = reconstruct_matrix_from_svd(s_k, U_full, V_full)
        np.testing.assert_allclose(C.numpy(), C_hat.numpy(), atol=1e-8)

    def test_singular_values_nonnegative(self):
        """Singular values must be non-negative."""
        tf.random.set_seed(42)
        C = tf.random.normal((3, 10, 5), dtype=tf.float64)
        s_k, _, _ = svd_via_eigh_full(C)
        self.assertTrue(np.all(s_k.numpy() >= 0))

    def test_orthonormality_U(self):
        """U_full columns should be orthonormal."""
        tf.random.set_seed(42)
        C = tf.random.normal((2, 7, 9), dtype=tf.float64)
        _, U_full, _ = svd_via_eigh_full(C)
        UtU = tf.matmul(U_full, U_full, transpose_a=True)
        eye = tf.eye(tf.shape(U_full)[2], batch_shape=[2], dtype=tf.float64)
        np.testing.assert_allclose(UtU.numpy(), eye.numpy(), atol=1e-8)

    def test_orthonormality_V(self):
        """V_full columns should be orthonormal."""
        tf.random.set_seed(42)
        C = tf.random.normal((2, 5, 8), dtype=tf.float64)
        _, _, V_full = svd_via_eigh_full(C)
        VtV = tf.matmul(V_full, V_full, transpose_a=True)
        eye = tf.eye(tf.shape(V_full)[2], batch_shape=[2], dtype=tf.float64)
        np.testing.assert_allclose(VtV.numpy(), eye.numpy(), atol=1e-8)

    def test_square_matrix(self):
        """Should handle square matrices (n == m)."""
        tf.random.set_seed(42)
        C = tf.random.normal((3, 6, 6), dtype=tf.float64)
        s_k, U_full, V_full = svd_via_eigh_full(C)
        C_hat = reconstruct_matrix_from_svd(s_k, U_full, V_full)
        np.testing.assert_allclose(C.numpy(), C_hat.numpy(), atol=1e-8)

    def test_tall_matrix(self):
        """Should handle tall matrices (n > m)."""
        tf.random.set_seed(42)
        C = tf.random.normal((2, 12, 4), dtype=tf.float64)
        s_k, U_full, V_full = svd_via_eigh_full(C)
        self.assertEqual(s_k.shape[1], 4)
        C_hat = reconstruct_matrix_from_svd(s_k, U_full, V_full)
        np.testing.assert_allclose(C.numpy(), C_hat.numpy(), atol=1e-8)


# ============================================================================
# SVD graph-mode gradient tests (tf.linalg.qr stop_gradient fix)
# ============================================================================

class TestSvdGradientGraphMode(unittest.TestCase):
    """
    Tests that gradients through svd_via_eigh_full work in graph mode
    with dynamic shapes (the original NotImplementedError scenario).
    """

    def test_gradient_exists_graph_mode(self):
        """Gradient through s_k must be non-None in @tf.function."""

        @tf.function
        def compute_grad(C):
            with tf.GradientTape() as tape:
                tape.watch(C)
                s_k, _, _ = svd_via_eigh_full(C)
                loss = tf.reduce_sum(s_k)
            return tape.gradient(loss, C)

        tf.random.set_seed(42)
        C = tf.random.normal((2, 5, 8), dtype=tf.float64)
        grad = compute_grad(C)
        self.assertIsNotNone(grad)
        self.assertFalse(np.any(np.isnan(grad.numpy())))

    def test_gradient_varying_shapes_graph_mode(self):
        """Single @tf.function handles multiple dynamic shapes without error."""

        @tf.function(reduce_retracing=True)
        def compute_grad(C):
            with tf.GradientTape() as tape:
                tape.watch(C)
                s_k, _, _ = svd_via_eigh_full(C)
                loss = tf.reduce_sum(s_k ** 2)
            return tape.gradient(loss, C)

        for n, m in [(5, 8), (10, 3), (7, 7), (20, 15)]:
            tf.random.set_seed(42)
            C = tf.random.normal((2, n, m), dtype=tf.float64)
            grad = compute_grad(C)
            self.assertEqual(grad.shape, C.shape,
                             f"Gradient shape mismatch for n={n}, m={m}")
            self.assertFalse(np.any(np.isnan(grad.numpy())),
                             f"NaN gradient for n={n}, m={m}")

    def test_gradient_through_reconstruction(self):
        """Gradient flows through full SVD -> reconstruct pipeline."""

        @tf.function
        def compute_grad(C):
            with tf.GradientTape() as tape:
                tape.watch(C)
                s_k, U_full, V_full = svd_via_eigh_full(C)
                C_hat = reconstruct_matrix_from_svd(s_k, U_full, V_full)
                loss = tf.reduce_sum(C_hat ** 2)
            return tape.gradient(loss, C)

        tf.random.set_seed(42)
        C = tf.random.normal((3, 6, 9), dtype=tf.float64)
        grad = compute_grad(C)
        self.assertIsNotNone(grad)
        self.assertFalse(np.any(np.isnan(grad.numpy())))


# ============================================================================
# End-to-end CrossRIELayer gradient tests
# ============================================================================

class TestCrossRIELayerGradient(unittest.TestCase):
    """
    End-to-end gradient tests for the full CrossRIELayer in graph mode
    with dynamic matrix dimensions.
    """

    def _build_model(self, **layer_kwargs):
        layer = CrossRIELayer(
            encoding_units=[8], lstm_units=[16],
            final_hidden_layer_sizes=[8], **layer_kwargs
        )
        in_Cxx = tf.keras.Input(shape=(None, None))
        in_Cyy = tf.keras.Input(shape=(None, None))
        in_Cxy = tf.keras.Input(shape=(None, None))
        in_t = tf.keras.Input(shape=())
        out = layer([in_Cxx, in_Cyy, in_Cxy, in_t])
        model = tf.keras.Model(inputs=[in_Cxx, in_Cyy, in_Cxy, in_t], outputs=out)
        model.compile(optimizer='adam', loss='mse')
        return model

    def _make_batch(self, B, N, M, T):
        tf.random.set_seed(42)
        Cxx = tf.random.normal((B, N, N), dtype=tf.float64)
        Cyy = tf.random.normal((B, M, M), dtype=tf.float64)
        Cxy = tf.random.normal((B, N, M), dtype=tf.float64)
        T_s = tf.constant([float(T)] * B, dtype=tf.float64)
        target = tf.random.normal((B, N, M), dtype=tf.float64)
        return [Cxx, Cyy, Cxy, T_s], target

    def test_train_step_additive(self):
        """Single train_on_batch in additive mode succeeds."""
        model = self._build_model(multiplicative=False, final_activation='linear')
        inputs, target = self._make_batch(4, 8, 10, 100)
        loss = model.train_on_batch(inputs, target)
        self.assertGreater(loss, 0)

    def test_train_step_multiplicative(self):
        """Single train_on_batch in multiplicative mode succeeds."""
        model = self._build_model(multiplicative=True, final_activation='softplus')
        inputs, target = self._make_batch(4, 8, 10, 100)
        loss = model.train_on_batch(inputs, target)
        self.assertGreater(loss, 0)

    def test_varying_shapes_sequential_training(self):
        """Model trains on batches with different N, M without errors."""
        model = self._build_model(multiplicative=False, final_activation='linear')
        shapes = [(6, 10), (12, 5), (8, 8), (4, 15)]
        for N, M in shapes:
            inputs, target = self._make_batch(4, N, M, 100)
            loss = model.train_on_batch(inputs, target)
            self.assertGreater(loss, 0, f"Zero loss for N={N}, M={M}")

    def test_loss_decreases(self):
        """Loss decreases over multiple training steps (basic sanity)."""
        model = self._build_model(multiplicative=False, final_activation='linear')
        inputs, target = self._make_batch(8, 6, 8, 100)
        losses = []
        for _ in range(10):
            loss = model.train_on_batch(inputs, target)
            losses.append(loss)
        self.assertLess(losses[-1], losses[0],
                        "Loss did not decrease over 10 steps on the same batch")


if __name__ == '__main__':
    unittest.main()
