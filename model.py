import tensorflow as tf
from keras import layers, Model
try:
    from . import utils as cl
    from . import utils as ccf
except ImportError:
    import utils as cl
    import utils as ccf

def GeneralizedCCCModel(encoding_units, lstm_units, final_hidden_layer_sizes,
                        multiplicative, final_activation, 
                        use_raw_returns=False, outputs=['Cxy'],**kwargs):
    len(kwargs) > 0 and print(f"Warning: model received unused arguments: {list(kwargs.keys())}")
    
    if multiplicative and final_activation not in ['softplus', 'relu', 'sigmoid']:
        raise ValueError("For multiplicative models, final_activation must be one of 'softplus', 'relu', or 'sigmoid' to ensure non-negative outputs.")
    if not multiplicative and final_activation not in ['linear', 'tanh']:
        raise ValueError("For additive models, final_activation must be either 'linear' or 'tanh'.")
    
    if use_raw_returns:
        Rx = layers.Input(shape=(None, None), name='rx')  # (B, N, T)
        Ry = layers.Input(shape=(None, None), name='ry')  # (B, M, T)
        inputs = [Rx, Ry]

        zscore_x = cl.ZScoreLayer(name='ZScore_rx')(Rx)
        zscore_y = cl.ZScoreLayer(name='ZScore_ry')(Ry)

        sample_length_layer = cl.SeriesLengthLayer(name='SeriesLength')
        n_samples = sample_length_layer(Rx)
        safe_counts = cl.SafeSampleCountLayer(name='SafeSamples')(n_samples)
        denom = layers.Reshape((1, 1), name='SampleDenominatorReshape')(safe_counts)

        Cxx = cl.CovarianceLayer(normalize=True, name='Cov_xx_from_rx')(zscore_x)
        Cyy = cl.CovarianceLayer(normalize=True, name='Cov_yy_from_ry')(zscore_y)
        Cxy = cl.CrossCovarianceLayer(name='CrossCorrelation')([zscore_x, zscore_y, denom])
    else:
        Cxx = layers.Input(shape=(None, None))  # (B, N, N)
        Cyy = layers.Input(shape=(None, None)) # (B, M, M)
        Cxy = layers.Input(shape=(None, None)) # (B, N, M)
        n_samples = layers.Input(shape=(), dtype=tf.float32) # (B,)
        inputs = [Cxx, Cyy, Cxy, n_samples]
    # SVD decomposition of cross-correlation matrix
    Sxy, Lxy, Rxy = ccf.SVDViaEighFullLayer(name='SVD_Cxy')(Cxy)
    # Compute diagonal elements
    Pxx = ccf.DiagVtCvLayer(name='DiagVtCv_xx')([Cxx, Lxy])
    Pyy = ccf.DiagVtCvLayer(name='DiagVtCv_yy')([Cyy, Rxy])
    # Add dimension awareness
    Sxy_expanded = cl.ExpandDimsLayer()(Sxy) # (B, N, 1)
    Pxx_expanded = ccf.DimAware2DLayer(name='DimAware2D_xx',features=['q1'])([Pxx, Cxy, n_samples]) # (B, N, 2)
    Pyy_expanded = ccf.DimAware2DLayer(name='DimAware2D_yy',features=['q2'])([Pyy, Cxy, n_samples]) # (B, M, 2)
    Sxy_padded = ccf.PadA2BSimpleLayer(name='Pad_xy')([Sxy_expanded, Cxy]) # (B, M, 1)
    Pxx_padded = ccf.PadA2BSimpleLayer(name='Pad_xx')([Pxx_expanded, Cxy]) # (B, M, 2)
    Pyy_padded = ccf.PadA2BSimpleLayer(name='Pad_yy')([Pyy_expanded, Cxy]) # (B, M, 2)
    Pxx_Sxy = layers.Concatenate()([Pxx_padded, Sxy_padded])
    Pyy_Sxy = layers.Concatenate()([Pyy_padded, Sxy_padded])
    if encoding_units:
        encoder = cl.DeepLayer(hidden_layer_sizes=encoding_units,name='Encoder_Deep')
        Tokens = encoder(Pxx_Sxy) + encoder(Pyy_Sxy) # (B, M, 4)
    else:
        Tokens = Pxx_Sxy + Pyy_Sxy
    Shrinkage = cl.DeepRecurrentLayer(recurrent_layer_sizes=lstm_units, final_hidden_layer_sizes=final_hidden_layer_sizes,
                                        final_activation=final_activation, name='DeepRecurrent')(Tokens)
    Shrinkage = cl.TakeTop()([Shrinkage, Sxy])
    if multiplicative:
        Sxy_cleaned = Shrinkage * Sxy
    else:
        Sxy_cleaned = Shrinkage + Sxy
        
    outputs_dict = {'Sxy': Sxy_cleaned}
    if 'Cxy' in outputs:
        Cxy_denoised = ccf.SVDReconstructFromFullLayer(name='SVD_Reconstruct')([Sxy_cleaned, Lxy, Rxy])
        outputs_dict['Cxy'] = Cxy_denoised
    
        
    if len(outputs) == 1:
        model = Model(inputs=inputs, outputs=outputs_dict[outputs[0]])
    else:
        model = Model(inputs=inputs, outputs=outputs_dict)
        
    return model
