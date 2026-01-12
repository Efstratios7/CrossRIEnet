import tensorflow as tf
from keras import layers, Model
from . import custom_layers as cl
from typing import Optional, List, Any

@tf.keras.utils.register_keras_serializable(package='CCC_Models', name='GeneralizedCCCLayer')
class GeneralizedCCCLayer(layers.Layer):
    """
    Generalized Cross-Correlation Correction (CCC) Layer.
    Uses deep learning to denoise cross-correlation matrices.

    Args:
        encoding_units: List of integers for the encoder dense layers.
        lstm_units: List of integers for the LSTM layers.
        final_hidden_layer_sizes: List of integers for the final dense layers.
        multiplicative: If True, applies multiplicative shrinkage (for non-negative outputs).
        final_activation: Activation function for the final output ('softplus', 'relu', 'sigmoid', 'linear', 'tanh').
        outputs: List of keys specifying which outputs to return (default ['Cxy']).
        **kwargs: Standard Keras Layer arguments.

    Returns:
         Denoised cross-correlation matrix tensor (or dictionary if multiple outputs selected).
    """
    def __init__(self, 
                 encoding_units: List[int], 
                 lstm_units: List[int], 
                 final_hidden_layer_sizes: List[int],
                 multiplicative: bool, 
                 final_activation: str, 
                 outputs: List[str] = ['Cxy'], 
                 **kwargs):
        super(GeneralizedCCCLayer, self).__init__(**kwargs)
        
        if multiplicative and final_activation not in ['softplus', 'relu', 'sigmoid']:
            raise ValueError("For multiplicative models, final_activation must be one of 'softplus', 'relu', or 'sigmoid' to ensure non-negative outputs.")
        if not multiplicative and final_activation not in ['linear', 'tanh']:
            raise ValueError("For additive models, final_activation must be either 'linear' or 'tanh'.")
        
        self.encoding_units = encoding_units
        self.lstm_units = lstm_units
        self.final_hidden_layer_sizes = final_hidden_layer_sizes
        self.multiplicative = multiplicative
        self.final_activation = final_activation
        self.outputs_keys = outputs

        # Components
        self.svd_layer = cl.SpectralSVDLayer(name='SVD_Cxy')
        self.diag_xx = cl.ProjectedVarianceDiagonalLayer(name='DiagVtCv_xx')
        self.diag_yy = cl.ProjectedVarianceDiagonalLayer(name='DiagVtCv_yy')
        self.expand_dims = cl.ExpandDimsLayer()
        self.dim_aware_xx = cl.DimensionAwarenessLayer(name='DimAware2D_xx', features=['q1'])
        self.dim_aware_yy = cl.DimensionAwarenessLayer(name='DimAware2D_yy', features=['q2'])
        self.pad_xy = cl.DimensionMatchingLayer(name='Pad_xy')
        self.pad_xx = cl.DimensionMatchingLayer(name='Pad_xx')
        self.pad_yy = cl.DimensionMatchingLayer(name='Pad_yy')
        self.concat = layers.Concatenate()
        
        if self.encoding_units:
            self.encoder = cl.DeepLayer(hidden_layer_sizes=self.encoding_units, name='Encoder_Deep')
        else:
            self.encoder = None

        self.shrinkage = cl.DeepRecurrentLayer(
            recurrent_layer_sizes=self.lstm_units, 
            final_hidden_layer_sizes=self.final_hidden_layer_sizes,
            final_activation=self.final_activation, 
            name='DeepRecurrent'
        )
        self.take_top = cl.TakeTop()
        self.svd_recon = cl.SVDReconstructionLayer(name='SVD_Reconstruct')

    def call(self, inputs: List[Any]) -> Any:
        """
        Forward pass of the CCC model.

        Args:
            inputs: List containing [Cxx, Cyy, Cxy, n_samples].
            Cxx: Cross-covariance matrix of variable 1. [Batch, N, N]
            Cyy: Cross-covariance matrix of variable 2. [Batch, M, M]
            Cxy: Cross-correlation matrix of variable 1 and variable 2. [Batch, N, M]
            n_samples: Number of samples. [Batch]

        Returns:
            Denoised Cxy (and optionally Sxy) depending on configuration. [Batch, N, M]
        """
        Cxx, Cyy, Cxy, n_samples = inputs
        
        # SVD decomposition of cross-correlation matrix
        Sxy, Lxy, Rxy = self.svd_layer(Cxy)
        
        # Compute diagonal elements
        Pxx = self.diag_xx([Cxx, Lxy])
        Pyy = self.diag_yy([Cyy, Rxy])
        
        # Add dimension awareness
        Sxy_expanded = self.expand_dims(Sxy) # (B, N, 1)
        Pxx_expanded = self.dim_aware_xx([Pxx, Cxy, n_samples]) # (B, N, 2)
        Pyy_expanded = self.dim_aware_yy([Pyy, Cxy, n_samples]) # (B, M, 2)
        
        Sxy_padded = self.pad_xy([Sxy_expanded, Cxy]) # (B, M, 1)
        Pxx_padded = self.pad_xx([Pxx_expanded, Cxy]) # (B, M, 2)
        Pyy_padded = self.pad_yy([Pyy_expanded, Cxy]) # (B, M, 2)
        
        Pxx_Sxy = self.concat([Pxx_padded, Sxy_padded])
        Pyy_Sxy = self.concat([Pyy_padded, Sxy_padded])
        
        if self.encoder:
            Tokens = self.encoder(Pxx_Sxy) + self.encoder(Pyy_Sxy) # (B, M, 4)
        else:
            Tokens = Pxx_Sxy + Pyy_Sxy
            
        Shrinkage = self.shrinkage(Tokens)
        Shrinkage = self.take_top([Shrinkage, Sxy])
        
        if self.multiplicative:
            Sxy_cleaned = Shrinkage * Sxy
        else:
            Sxy_cleaned = Shrinkage + Sxy
            
        outputs_dict = {'Sxy': Sxy_cleaned}
        if 'Cxy' in self.outputs_keys:
            Cxy_denoised = self.svd_recon([Sxy_cleaned, Lxy, Rxy])
            outputs_dict['Cxy'] = Cxy_denoised
        
        if len(self.outputs_keys) == 1:
            return outputs_dict[self.outputs_keys[0]]
        else:
            return outputs_dict

    def get_config(self):
        config = super(GeneralizedCCCLayer, self).get_config()
        config.update({
            'encoding_units': self.encoding_units,
            'lstm_units': self.lstm_units,
            'final_hidden_layer_sizes': self.final_hidden_layer_sizes,
            'multiplicative': self.multiplicative,
            'final_activation': self.final_activation,
            'outputs': self.outputs_keys
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

