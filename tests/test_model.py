import unittest
import sys
import os
import tensorflow as tf
from keras import losses, optimizers
import numpy as np
import random

from ccc import GeneralizedCCCModel

class TestGeneralizedCCCModel(unittest.TestCase):
    def setUp(self):
        # Default baseline parameters
        self.B = 2
        self.N = 5
        self.M = 10
        self.T = 100
        self.encoding_units = [8]
        self.lstm_units = [16]
        self.final_hidden_layer_sizes = [8]
        
    def test_model_initialization_additive(self):
        """Test instantiation and forward pass in Additive mode"""
        model = GeneralizedCCCModel(
            encoding_units=self.encoding_units,
            lstm_units=self.lstm_units,
            final_hidden_layer_sizes=self.final_hidden_layer_sizes,
            multiplicative=False,
            final_activation='linear',

        )
        
        Cxx = tf.random.normal((self.B, self.N, self.N))
        Cyy = tf.random.normal((self.B, self.M, self.M))
        Cxy = tf.random.normal((self.B, self.N, self.M))
        n_samples = tf.constant([100.0] * self.B)
        
        output = model([Cxx, Cyy, Cxy, n_samples])
        
        # Output should be (B, N, M) - corresponding to Cxy shape
        self.assertEqual(output.shape, (self.B, self.N, self.M))

    def test_model_initialization_multiplicative(self):
        """Test instantiation and forward pass in Multiplicative mode"""
        model = GeneralizedCCCModel(
            encoding_units=self.encoding_units,
            lstm_units=self.lstm_units,
            final_hidden_layer_sizes=self.final_hidden_layer_sizes,
            multiplicative=True,
            final_activation='softplus',

        )
        
        Cxx = tf.random.normal((self.B, self.N, self.N))
        Cyy = tf.random.normal((self.B, self.M, self.M))
        Cxy = tf.random.normal((self.B, self.N, self.M))
        n_samples = tf.constant([100.0] * self.B)
        
        output = model([Cxx, Cyy, Cxy, n_samples])
        self.assertEqual(output.shape, (self.B, self.N, self.M))



    def test_varying_architectures(self):
        """Test different network depths and widths"""
        configs = [
            {'encoding': [32, 16], 'lstm': [64], 'final': [32]},
            {'encoding': [], 'lstm': [16], 'final': []}, # Minimal
            {'encoding': [8], 'lstm': [8, 8], 'final': [4]} # Deep recurrent
        ]
        
        for conf in configs:
            with self.subTest(config=conf):
                model = GeneralizedCCCModel(
                    encoding_units=conf['encoding'],
                    lstm_units=conf['lstm'],
                    final_hidden_layer_sizes=conf['final'],
                    multiplicative=True,
                    final_activation='softplus'
                )
                
                Cxx = tf.random.normal((self.B, self.N, self.N))
                Cyy = tf.random.normal((self.B, self.M, self.M))
                Cxy = tf.random.normal((self.B, self.N, self.M))
                n_samples = tf.constant([100.0] * self.B)
                
                output = model([Cxx, Cyy, Cxy, n_samples])
                self.assertEqual(output.shape, (self.B, self.N, self.M))

    def test_valid_activations(self):
        """Test all supported activation functions for multiplicative mode"""
        activations = ['softplus', 'relu', 'sigmoid']
        for act in activations:
            with self.subTest(activation=act):
                model = GeneralizedCCCModel(
                    encoding_units=self.encoding_units,
                    lstm_units=self.lstm_units,
                    final_hidden_layer_sizes=self.final_hidden_layer_sizes,
                    multiplicative=True,
                    final_activation=act
                )
                # Just verify it builds and runs without error
                Cxx = tf.random.normal((self.B, self.N, self.N))
                Cyy = tf.random.normal((self.B, self.M, self.M))
                Cxy = tf.random.normal((self.B, self.N, self.M))
                n_samples = tf.constant([100.0] * self.B)
                model([Cxx, Cyy, Cxy, n_samples])

    def test_additive_activations(self):
        """Test supported activations for additive mode"""
        activations = ['linear', 'tanh']
        for act in activations:
             with self.subTest(activation=act):
                model = GeneralizedCCCModel(
                    encoding_units=self.encoding_units,
                    lstm_units=self.lstm_units,
                    final_hidden_layer_sizes=self.final_hidden_layer_sizes,
                    multiplicative=False,
                    final_activation=act
                )
                Cxx = tf.random.normal((self.B, self.N, self.N))
                Cyy = tf.random.normal((self.B, self.M, self.M))
                Cxy = tf.random.normal((self.B, self.N, self.M))
                n_samples = tf.constant([100.0] * self.B)
                model([Cxx, Cyy, Cxy, n_samples])

    def test_invalid_activation_config(self):
        """Test validation logic for activation functions"""
        with self.assertRaises(ValueError):
            GeneralizedCCCModel(
                encoding_units=self.encoding_units,
                lstm_units=self.lstm_units,
                final_hidden_layer_sizes=self.final_hidden_layer_sizes,
                multiplicative=True,
                final_activation='linear', # Invalid for multiplicative

            )

    def test_large_dimensions(self):
        """Stress test with slightly larger matrices"""
        N_large = 50
        M_large = 60
        model = GeneralizedCCCModel(
            encoding_units=[16],
            lstm_units=[16],
            final_hidden_layer_sizes=[16],
            multiplicative=True,
            final_activation='softplus'
        )
        Cxx = tf.random.normal((self.B, N_large, N_large))
        Cyy = tf.random.normal((self.B, M_large, M_large))
        Cxy = tf.random.normal((self.B, N_large, M_large))
        n_samples = tf.constant([200.0] * self.B)
        
        output = model([Cxx, Cyy, Cxy, n_samples])
        self.assertEqual(output.shape, (self.B, N_large, M_large))


    def test_training_simulation(self):
        """Simulate a user training the model with random data"""
        # 1. Setup Data
        B, N, M = 4, 10, 10
        Cxx = tf.random.normal((B, N, N))
        Cyy = tf.random.normal((B, M, M))
        Cxy_noisy = tf.random.normal((B, N, M))
        n_samples = tf.constant([100.0] * B)
        
        # Target (Clean Cxy)
        Cxy_clean = tf.random.normal((B, N, M))
        
        # 2. Initialize Model
        model = GeneralizedCCCModel(
            encoding_units=[16],
            lstm_units=[16],
            final_hidden_layer_sizes=[8],
            multiplicative=True,
            final_activation='softplus'
        )
        
        # 3. Compile
        optimizer = optimizers.Adam(learning_rate=1e-3)
        loss_fn = losses.MeanSquaredError()
        model.compile(optimizer=optimizer, loss=loss_fn)
        
        # 4. Train for one step
        # Note: input is [Cxx, Cyy, Cxy, n_samples]
        history = model.fit(
            x=[Cxx, Cyy, Cxy_noisy, n_samples],
            y=Cxy_clean,
            epochs=1,
            batch_size=2,
            verbose=0
        )
        
        # 5. Check loss exists
        loss = history.history['loss'][0]
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0.0)
        
        # 6. Check forward pass after training
        pred = model.predict([Cxx, Cyy, Cxy_noisy, n_samples])
        self.assertEqual(pred.shape, (B, N, M))


    def test_training_variable_dimensions(self):
        """Simulate training with random variable dimensions (Days [200,600], Stocks [100,250])"""
        # Random dimensions
        ndays = random.randint(200, 600)
        nstocks_N = random.randint(100, 250)
        nstocks_M = random.randint(100, 250)
        
        # print(f"\nRunning variable dimension test with: T={ndays}, N={nstocks_N}, M={nstocks_M}")
        
        # 1. Setup Data
        B = 2
        Cxx = tf.random.normal((B, nstocks_N, nstocks_N))
        Cyy = tf.random.normal((B, nstocks_M, nstocks_M))
        Cxy_noisy = tf.random.normal((B, nstocks_N, nstocks_M))
        n_samples = tf.constant([float(ndays)] * B)
        
        # Target
        Cxy_clean = tf.random.normal((B, nstocks_N, nstocks_M))
        
        # 2. Initialize Model
        model = GeneralizedCCCModel(
            encoding_units=[32],
            lstm_units=[16],
            final_hidden_layer_sizes=[16],
            multiplicative=True,
            final_activation='softplus'
        )
        
        # 3. Compile
        model.compile(optimizer='adam', loss='mse')
        
        # 4. Train
        history = model.fit(
            x=[Cxx, Cyy, Cxy_noisy, n_samples],
            y=Cxy_clean,
            epochs=1,
            batch_size=B,
            verbose=0
        )
        
        loss = history.history['loss'][0]
        self.assertGreater(loss, 0.0)
        
        # 5. Predict
        pred = model.predict([Cxx, Cyy, Cxy_noisy, n_samples])
        self.assertEqual(pred.shape, (B, nstocks_N, nstocks_M))

if __name__ == '__main__':
    unittest.main()
