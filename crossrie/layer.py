import tensorflow as tf
from keras import layers, Model
from . import custom_layers as cl
from typing import Optional, List, Any

@tf.keras.utils.register_keras_serializable(package='crossrie', name='CrossRIELayer')
class CrossRIELayer(layers.Layer):
    """
    Generalized Cross-Correlation Correction (CrossRIE) Layer.
    Uses deep learning to denoise cross-correlation matrices.

    Args:
        encoding_units: List of integers for the shared encoder (E_theta) MLP structure. e.g. [16, 2] for a 16-unit hidden layer and 2D embedding.
        lstm_units: List of integers for the bidirectional LSTM aggregator hidden sizes. e.g. [128, 64].
        final_hidden_layer_sizes: List of integers for the pointwise head (g_theta) hidden layers. e.g. [252] for a 252-unit hidden layer. One can also use a list e.g. [256,64] which turns it into a Multilayer Perceptron with more depth.
        multiplicative: If True, applies bounded multiplicative correction (s_tilde = s_hat * sigma(delta)) which enforces non-negativity. If False, applies additive correction (s_tilde = s_hat + delta) which is preferred in experiments.
        final_activation: Activation function for the scalar correction delta. For additive mode, 'linear' is default. For multiplicative mode, 'sigmoid' is used for bounded correction.
        outputs: List of keys specifying which outputs to return (default ['Cxy']). e.g. ['Cxy', 'Sxy']
        **kwargs: Standard Keras Layer arguments.

    Returns:
         Denoised cross-correlation matrix tensor (or dictionary if multiple outputs selected). If Sxy is selected, it returns the learned singular values.
    """
    def __init__(self, 
                 encoding_units: List[int] = [16,2], 
                 lstm_units: List[int] = [128,64], 
                 final_hidden_layer_sizes: List[int] = [252],
                 multiplicative: bool = False, 
                 final_activation: str = 'linear', 
                 outputs: List[str] = ['Cxy'], 
                 **kwargs):
        super(CrossRIELayer, self).__init__(**kwargs)
        
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
        


        self.two_stream_encoder = cl.Two_Stream_EncoderLayer(
            encoding_units=self.encoding_units,
            lstm_units=self.lstm_units,
            final_hidden_layer_sizes=self.final_hidden_layer_sizes,
            final_activation=self.final_activation,
            name='TwoStreamEncoder'
        )
        self.take_top = cl.TakeTop()
        self.svd_recon = cl.SVDReconstructionLayer(name='SVD_Reconstruct')

    def build(self, input_shape):
        """
        Builds the layer and its sub-layers.
        
        Args:
            input_shape: List of input shapes: [Cxx_shape, Cyy_shape, Cxy_shape, T_samples_shape].
        """
        # Calculate feature dimension for encoder input
        # Pxx_expanded has 1 + len(dim_aware_xx.features) channels
        # Sxy_padded has 1 channel
        channels_xx = 1 + len(self.dim_aware_xx.features)
        channels_yy = 1 + len(self.dim_aware_yy.features)
        
        if channels_xx != channels_yy and self.encoding_units:
            raise ValueError(f"Channel mismatch for shared encoder: channels_xx={channels_xx}, channels_yy={channels_yy}. Both streams must have same number of features.")
        
        total_channels_xx = channels_xx + 1 # +1 for Sxy
        total_channels_yy = channels_yy + 1 # +1 for Sxy

        # Build encoder
        # Input shape: (Batch, SequenceLength, Features)
        # Pxx_Sxy has shape (Batch, M, total_channels_xx)
        self.two_stream_encoder.build([(None, None, total_channels_xx), (None, None, total_channels_yy)])
        
        super(CrossRIELayer, self).build(input_shape)

    def call(self, inputs: List[Any]) -> Any:
        """
        Forward pass of the cross-covariance cleaning model.

        Args:
            inputs: List containing [Cxx, Cyy, Cxy, T_samples].
            Cxx: Cross-covariance matrix of variable 1. [Batch, N, N]
            Cyy: Cross-covariance matrix of variable 2. [Batch, M, M]
            Cxy: Cross-correlation matrix of variable 1 and variable 2. [Batch, N, M]
            T_samples: Number of samples. [Batch]

        Returns:
            Denoised Cxy (and optionally Sxy) depending on configuration. [Batch, N, M]
        """
        Cxx, Cyy, Cxy, T_samples = inputs
        
        # SVD decomposition of cross-correlation matrix
        Sxy, Lxy, Rxy = self.svd_layer(Cxy)
        
        # Compute diagonal elements
        Pxx = self.diag_xx([Cxx, Lxy])
        Pyy = self.diag_yy([Cyy, Rxy])
        
        # Add dimension awareness
        Sxy_expanded = self.expand_dims(Sxy) # (B, N, 1)
        Pxx_expanded = self.dim_aware_xx([Pxx, Cxy, T_samples]) # (B, N, 2)
        Pyy_expanded = self.dim_aware_yy([Pyy, Cxy, T_samples]) # (B, M, 2)
        
        Sxy_padded = self.pad_xy([Sxy_expanded, Cxy]) # (B, M, 1)
        Pxx_padded = self.pad_xx([Pxx_expanded, Cxy]) # (B, M, 2)
        Pyy_padded = self.pad_yy([Pyy_expanded, Cxy]) # (B, M, 2)
        
        Pxx_Sxy = self.concat([Pxx_padded, Sxy_padded])
        Pyy_Sxy = self.concat([Pyy_padded, Sxy_padded])
        
        aggregator_head = self.two_stream_encoder([Pxx_Sxy, Pyy_Sxy])
        aggregated_head = self.take_top([aggregator_head, Sxy])
        
        if self.multiplicative:
            Sxy_cleaned = aggregated_head * Sxy
        else:
            Sxy_cleaned = aggregated_head + Sxy
            
        outputs_dict = {'Sxy': Sxy_cleaned}
        if 'Cxy' in self.outputs_keys:
            Cxy_denoised = self.svd_recon([Sxy_cleaned, Lxy, Rxy])
            outputs_dict['Cxy'] = Cxy_denoised
        
        if len(self.outputs_keys) == 1:
            return outputs_dict[self.outputs_keys[0]]
        else:
            return outputs_dict

    def get_config(self):
        config = super(CrossRIELayer, self).get_config()
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

