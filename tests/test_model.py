import unittest
import sys
import os

# Suppress TensorFlow C++ backend logs for cleaner test output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from keras import losses, optimizers
from crossrie import CrossRIELayer

class TestCrossRIELayer(unittest.TestCase):
    def _run_dynamic_training_test(self, layer, N_range=(20, 50), M_range=(20, 50)):
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from tests.data_generator import get_dynamic_dataset

        # Define Inputs EXPLICITLY with abstract `None` shapes.
        in_Cxx = tf.keras.Input(shape=(None, None))
        in_Cyy = tf.keras.Input(shape=(None, None))
        in_Cxy = tf.keras.Input(shape=(None, None))
        in_n = tf.keras.Input(shape=())
        
        # Build Model
        out = layer([in_Cxx, in_Cyy, in_Cxy, in_n])
        model = tf.keras.Model(inputs=[in_Cxx, in_Cyy, in_Cxy, in_n], outputs=out)
        
        optimizer = optimizers.Adam(learning_rate=1e-3)
        loss_fn = losses.MeanSquaredError()
        model.compile(optimizer=optimizer, loss=loss_fn)
        
        # Pull Dynamic Dataset Pipeline
        dataset = get_dynamic_dataset(batch_size=16, N_range=N_range, M_range=M_range)
        
        history = model.fit(
            dataset,
            epochs=10,
            steps_per_epoch=21,
            verbose=0
        )
        
        self.assertEqual(len(history.history['loss']), 10)
        self.assertGreater(history.history['loss'][-1], 0.0)
        
        pred_data = next(iter(dataset.take(1)))
        inputs_pred, _ = pred_data
        
        pred = model.predict(inputs_pred, verbose=0)
        
        expected_shape = inputs_pred[0].shape
        self.assertEqual(pred.shape, (16, expected_shape[1], inputs_pred[1].shape[1]))

    def test_static_model_training(self):
        """Test simple training over 2 epochs utilizing exclusively static dimensions initialized uniformly across graph and data."""
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from tests.data_generator import get_dynamic_dataset
        
        # 1. Initialize Layer
        layer = CrossRIELayer(
            encoding_units=[8], lstm_units=[16], final_hidden_layer_sizes=[8],
            multiplicative=True, final_activation='softplus'
        )
        
        # Lock exact dimensions
        b_size, N_static, M_static, T_static = 8, 25, 30, 200
        
        # Define Inputs with rigid explicit shapes locking completely (opposite of variable)
        in_Cxx = tf.keras.Input(shape=(N_static, N_static))
        in_Cyy = tf.keras.Input(shape=(M_static, M_static))
        in_Cxy = tf.keras.Input(shape=(N_static, M_static))
        in_n = tf.keras.Input(shape=())
        
        # Build Model
        out = layer([in_Cxx, in_Cyy, in_Cxy, in_n])
        model = tf.keras.Model(inputs=[in_Cxx, in_Cyy, in_Cxy, in_n], outputs=out)
        
        model.compile(optimizer='adam', loss='mse')
        
        # Limit pipeline specifically to generate exactly these bounds
        dataset = get_dynamic_dataset(
            batch_size=b_size,
            N_range=(N_static, N_static),
            M_range=(M_static, M_static),
            ndays_range=(T_static, T_static)
        )
        
        # Evaluate for specifically 2 epochs
        history = model.fit(
            dataset,
            epochs=2,
            steps_per_epoch=5,
            verbose=0
        )
        
        self.assertEqual(len(history.history['loss']), 2)

    def test_model_initialization_additive(self):
        """Test instantiation and full robust training string in Additive mode"""
        layer = CrossRIELayer(
            encoding_units=[8], lstm_units=[16], final_hidden_layer_sizes=[8],
            multiplicative=False, final_activation='linear'
        )
        self._run_dynamic_training_test(layer)

    def test_model_initialization_multiplicative(self):
        """Test instantiation and full robust training string in Multiplicative mode"""
        layer = CrossRIELayer(
            encoding_units=[8], lstm_units=[16], final_hidden_layer_sizes=[8],
            multiplicative=True, final_activation='softplus'
        )
        self._run_dynamic_training_test(layer)

    def test_varying_architectures(self):
        """Test different network depths and widths across rigorous epochs"""
        configs = [
            {'encoding': [32, 16], 'lstm': [64], 'final': [32]},
            {'encoding': [], 'lstm': [16], 'final': []},
            {'encoding': [8], 'lstm': [8, 8], 'final': [4]}
        ]
        
        for conf in configs:
            with self.subTest(config=conf):
                layer = CrossRIELayer(
                    encoding_units=conf['encoding'], lstm_units=conf['lstm'], final_hidden_layer_sizes=conf['final'],
                    multiplicative=True, final_activation='softplus'
                )
                self._run_dynamic_training_test(layer)

    def test_valid_activations(self):
        """Test all supported activation functions continuously for multiplicative mode"""
        activations = ['softplus', 'relu', 'sigmoid']
        for act in activations:
            with self.subTest(activation=act):
                layer = CrossRIELayer(
                    encoding_units=[8], lstm_units=[16], final_hidden_layer_sizes=[8],
                    multiplicative=True, final_activation=act
                )
                self._run_dynamic_training_test(layer)

    def test_additive_activations(self):
        """Test all supported activation functions continuously for additive mode"""
        activations = ['linear', 'tanh']
        for act in activations:
             with self.subTest(activation=act):
                layer = CrossRIELayer(
                    encoding_units=[8], lstm_units=[16], final_hidden_layer_sizes=[8],
                    multiplicative=False, final_activation=act
                )
                self._run_dynamic_training_test(layer)

    def test_invalid_activation_config(self):
        """Test validation logic (static validation behavior unchanged)"""
        with self.assertRaises(ValueError):
            CrossRIELayer(
                multiplicative=True, final_activation='linear'
            )

    def test_large_dimensions(self):
        """Stress test with explicitly large variable matrix spaces natively passed into variable training logic."""
        layer = CrossRIELayer(
            encoding_units=[16], lstm_units=[16], final_hidden_layer_sizes=[16],
            multiplicative=True, final_activation='softplus'
        )
        self._run_dynamic_training_test(layer, N_range=(250, 300), M_range=(250, 300),ndays_range=(200,250))

    def test_training_simulation(self):
        """Formerly a test for 1 epoch, now identically verified natively through the standard high iteration handler."""
        layer = CrossRIELayer(
            encoding_units=[16], lstm_units=[16], final_hidden_layer_sizes=[8],
            multiplicative=True, final_activation='softplus'
        )
        self._run_dynamic_training_test(layer)

    def test_training_variable_dimensions(self):
        """The baseline variable dimension explicit trainer implementation, integrated comprehensively into _run_dynamic_training_test."""
        layer = CrossRIELayer(
            encoding_units=[32], lstm_units=[16], final_hidden_layer_sizes=[16],
            multiplicative=True, final_activation='softplus'
        )
        self._run_dynamic_training_test(layer)

if __name__ == '__main__':
    unittest.main()
