import tensorflow as tf
from keras import layers
try:
    from . import helper_functions as hf
except ImportError:
    import helper_functions as hf

# ============================================================================
# CUSTOM LAYERS
# ============================================================================

@tf.keras.utils.register_keras_serializable(package='Custom_Layers', name='ZScoreLayer')
class ZScoreLayer(layers.Layer):
    def __init__(self, epsilon=1e-6, name=None, **kwargs):
        if name is None:
            raise ValueError("ZScoreLayer must have a name.")
        super(ZScoreLayer, self).__init__(name=name, **kwargs)
        self.epsilon = float(epsilon)

    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        centered = inputs - mean
        variance = tf.reduce_mean(tf.square(centered), axis=-1, keepdims=True)
        std = tf.sqrt(tf.maximum(variance, self.epsilon))
        return centered / std

    def get_config(self):
        config = super(ZScoreLayer, self).get_config()
        config.update({'epsilon': self.epsilon})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package='Custom_Layers', name='SeriesLengthLayer')
class SeriesLengthLayer(layers.Layer):
    def __init__(self, dtype=tf.float32, name=None, **kwargs):
        if name is None:
            raise ValueError("SeriesLengthLayer must have a name.")
        super(SeriesLengthLayer, self).__init__(name=name, **kwargs)
        self.output_dtype = tf.dtypes.as_dtype(dtype)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        steps = tf.cast(tf.shape(inputs)[-1], self.output_dtype)
        ones = tf.ones((batch_size,), dtype=self.output_dtype)
        return ones * steps

    def get_config(self):
        config = super(SeriesLengthLayer, self).get_config()
        config.update({'dtype': self.output_dtype.name})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package='Custom_Layers', name='SafeSampleCountLayer')
class SafeSampleCountLayer(layers.Layer):
    def __init__(self, minimum=1.0, name=None, **kwargs):
        if name is None:
            raise ValueError("SafeSampleCountLayer must have a name.")
        super(SafeSampleCountLayer, self).__init__(name=name, **kwargs)
        self.minimum = float(minimum)

    def call(self, inputs):
        values = tf.cast(inputs, tf.float32)
        return tf.maximum(values, self.minimum)

    def get_config(self):
        config = super(SafeSampleCountLayer, self).get_config()
        config.update({'minimum': self.minimum})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package='Custom_Layers', name='CovarianceLayer')
class CovarianceLayer(layers.Layer):
    def __init__(self, expand_dims=False, normalize=True, name=None, **kwargs):
        if name is None:
            raise ValueError("CovarianceLayer must have a name.")
        super(CovarianceLayer, self).__init__(name=name, **kwargs)
        self.expand_dims = expand_dims
        self.normalize = normalize

    def call(self, Returns):
        dtype = Returns.dtype
        if self.normalize:
            sample_size = tf.cast(tf.shape(Returns)[-1], dtype)
            Covariance = tf.matmul(Returns, Returns, transpose_b=True) / sample_size
        else:
            Covariance = tf.matmul(Returns, Returns, transpose_b=True)
        if self.expand_dims:
            Covariance = tf.expand_dims(Covariance, axis=-3)
        return Covariance

    def get_config(self):
        config = super(CovarianceLayer, self).get_config()
        config.update({
            'expand_dims': self.expand_dims,
            'normalize': self.normalize
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package='Custom_Layers', name='CrossCovarianceLayer')
class CrossCovarianceLayer(layers.Layer):
    def __init__(self, name=None, **kwargs):
        if name is None:
            raise ValueError("CrossCovarianceLayer must have a name.")
        super(CrossCovarianceLayer, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        x, y, denom = inputs
        numer = tf.matmul(x, y, transpose_b=True)
        denom = tf.cast(denom, numer.dtype)
        return numer / denom

    def get_config(self):
        return super(CrossCovarianceLayer, self).get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package='Custom_Layers', name='ExpandDimsLayer')
class ExpandDimsLayer(layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super(ExpandDimsLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)

    def get_config(self):
        config = super(ExpandDimsLayer, self).get_config()
        config.update({
            'axis': self.axis
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        shape.insert(self.axis if self.axis >= 0 else len(shape) + self.axis + 1, 1)
        return tuple(shape)

@tf.keras.utils.register_keras_serializable(package='CCC_Functions', name='SVDViaEighFullLayer')
class SVDViaEighFullLayer(tf.keras.layers.Layer):
    def __init__(self, eps=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.eps = eps
    
    def call(self, C):
        return hf.svd_via_eigh_full(C, eps=self.eps)
    
    def get_config(self):
        config = super().get_config()
        config.update({'eps': self.eps})
        return config

@tf.keras.utils.register_keras_serializable(package='CCC_Functions', name='DiagVtCvLayer')
class DiagVtCvLayer(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
    
    def call(self, inputs):
        C, V = inputs
        return hf.diag_vtCv(C, V)
    
    def get_config(self):
        return super().get_config()

@tf.keras.utils.register_keras_serializable(package='CCC_Functions', name='DimAware2DLayer')
class DimAware2DLayer(tf.keras.layers.Layer):
    def __init__(self, features=['n1', 'n2', 'q1', 'q2'], name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        valid_keys = {'n1', 'n2', 'q1', 'q2', 't', 't1', 't2'} 
        for f in features:
            if f not in valid_keys:
                raise ValueError(f"Feature '{f}' non supportata. Usa: {valid_keys}")
        self.features = features

    def compute_output_shape(self, input_shape):
        if len(input_shape) < 2:
             raise ValueError("DimAware2DLayer richiede [Mat, Shape_mat, t]")

        mat_shape = input_shape[0]       
        
        if len(mat_shape) == 2: 
            channels = 1
            n_dim = mat_shape[1]
            return (mat_shape[0], n_dim, channels + len(self.features))
        
        elif len(mat_shape) == 3: 
            channels = mat_shape[-1]
            n_dim = mat_shape[1]
            return (mat_shape[0], n_dim, channels + len(self.features))
        
        else:
            raise ValueError(f"Shape di Mat inattesa: {mat_shape}. Atteso (Batch, N) o (Batch, N, C)")

    def call(self, inputs):
        Mat, Shape_mat, t = inputs
        
        if len(Mat.shape) == 2:
            Mat_expanded = tf.expand_dims(Mat, axis=-1)
        else:
            Mat_expanded = Mat

        shape_tensor = tf.shape(Shape_mat)
        
        n_dim = tf.cast(shape_tensor[1], Mat.dtype)
        m_dim = tf.cast(shape_tensor[2], Mat.dtype)
        t_val = tf.cast(t, Mat.dtype)

        batch_size = tf.shape(Mat_expanded)[0]
        rows = tf.shape(Mat_expanded)[1] 
        
        t_reshaped = tf.reshape(t_val, (-1, 1, 1)) 
        
        feature_map = {}
        
        if any(x in self.features for x in ['n1', 'q1']):
            vals = tf.reshape(n_dim, (1, 1, 1))
            feature_map['n1'] = tf.broadcast_to(vals, (batch_size, rows, 1))

        if any(x in self.features for x in ['n2', 'q2']):
            vals = tf.reshape(m_dim, (1, 1, 1))
            feature_map['n2'] = tf.broadcast_to(vals, (batch_size, rows, 1))

        if any(x in self.features for x in ['t', 't1', 't2', 'q1', 'q2']):
            feature_map['t'] = tf.broadcast_to(t_reshaped, (batch_size, rows, 1))
            feature_map['t1'] = feature_map['t']
            feature_map['t2'] = feature_map['t']
            
        if 'q1' in self.features:
            feature_map['q1'] = feature_map['n1'] / (feature_map['t'] + 1e-7)
            
        if 'q2' in self.features:
            feature_map['q2'] = feature_map['n2'] / (feature_map['t'] + 1e-7)

        tensors_to_concat = [Mat_expanded]
        for feat in self.features:
            tensors_to_concat.append(feature_map[feat])
            
        return tf.concat(tensors_to_concat, axis=-1)
    
    def get_config(self):
        config = super().get_config()
        config.update({'features': self.features})
        return config

@tf.keras.utils.register_keras_serializable(package='CCC_Functions', name='PadA2BSimpleLayer')
class PadA2BSimpleLayer(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
    
    def call(self, inputs):
        A, B = inputs
        return hf.pad_A_to_B_simple(A, B)
    
    def get_config(self):
        return super().get_config()

@tf.keras.utils.register_keras_serializable(package='Custom_Layers', name='DeepLayer')
class DeepLayer(layers.Layer):
    def __init__(self, hidden_layer_sizes, last_activation="linear",
                 activation="leaky_relu", other_biases=True, last_bias=True,
                 dropout_rate=0., kernel_initializer="glorot_uniform", name=None, **kwargs):
        if name is None:
            raise ValueError("DeepLayer must have a name.")
        super(DeepLayer, self).__init__(name=name, **kwargs)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.last_activation = last_activation
        self.other_biases = other_biases
        self.last_bias = last_bias
        self.dropout_rate = dropout_rate
        self.kernel_initializer = kernel_initializer

        self.hidden_layers = []
        self.dropouts = []
        for i, size in enumerate(self.hidden_layer_sizes[:-1]):
            layer_name = f"{self.name}_hidden_{i}"
            dropout_name = f"{self.name}_dropout_{i}"
            dense = layers.Dense(size,
                                 activation=self.activation,
                                 use_bias=self.other_biases,
                                 kernel_initializer=self.kernel_initializer,
                                 name=layer_name)
            dropout = layers.Dropout(self.dropout_rate, name=dropout_name)
            self.hidden_layers.append(dense)
            self.dropouts.append(dropout)

        self.final_dense = layers.Dense(self.hidden_layer_sizes[-1],
                                        use_bias=self.last_bias,
                                        activation=self.last_activation,
                                        kernel_initializer=self.kernel_initializer,
                                        name=f"{self.name}_output")

    def build(self, input_shape):
        for dense, dropout in zip(self.hidden_layers, self.dropouts):
            dense.build(input_shape)
            input_shape = dense.compute_output_shape(input_shape)
            dropout.build(input_shape)
        self.final_dense.build(input_shape)
        super(DeepLayer, self).build(input_shape)

    def call(self, inputs):
        x = inputs
        for dense, dropout in zip(self.hidden_layers, self.dropouts):
            x = dense(x)
            x = dropout(x)
        outputs = self.final_dense(x)
        return outputs

    def get_config(self):
        config = super(DeepLayer, self).get_config()
        config.update({
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'activation': self.activation,
            'last_activation': self.last_activation,
            'other_biases': self.other_biases,
            'last_bias': self.last_bias,
            'dropout_rate': self.dropout_rate,
            'kernel_initializer': self.kernel_initializer
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.hidden_layer_sizes[-1]
        return tuple(output_shape)

@tf.keras.utils.register_keras_serializable(package='Custom_Layers', name='CustomNormalizationLayer')
class CustomNormalizationLayer(layers.Layer):
    def __init__(self, mode='sum', axis=-2, name=None, **kwargs):
        if name is None:
            raise ValueError("CustomNormalizationLayer must have a name.")
        super(CustomNormalizationLayer, self).__init__(name=name, **kwargs)
        self.mode = mode
        self.axis = axis

    def call(self, x):
        n = tf.cast(tf.shape(x)[self.axis], dtype=tf.float32)
        if self.mode == 'sum':
            x = n * x / tf.reduce_sum(x, axis=self.axis, keepdims=True)
        elif self.mode == 'inverse':
            inv = tf.math.reciprocal(x)
            x = n * inv / tf.reduce_sum(inv, axis=self.axis, keepdims=True)
            x = tf.math.reciprocal(x)
        return x

    def get_config(self):
        config = super(CustomNormalizationLayer, self).get_config()
        config.update({
            'mode': self.mode,
            'axis': self.axis
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package='Custom_Layers', name='DeepRecurrentLayer')
class DeepRecurrentLayer(layers.Layer):
    def __init__(self, recurrent_layer_sizes,final_activation="softplus", final_hidden_layer_sizes=[], final_hidden_activation="leaky_relu",
                 direction='bidirectional', dropout=0.,recurrent_dropout=0.,recurrent_model='LSTM', normalize=None, bottleneck=1, name=None, **kwargs):
        if name is None:
            raise ValueError("DeepRecurrentLayer must have a name.")
        super(DeepRecurrentLayer, self).__init__(name=name, **kwargs)

        self.recurrent_layer_sizes = recurrent_layer_sizes
        self.final_activation = final_activation
        self.final_hidden_layer_sizes = final_hidden_layer_sizes
        self.final_hidden_activation = final_hidden_activation
        self.direction = direction
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.recurrent_model = recurrent_model
        self.bottleneck = bottleneck
        if normalize not in [None, 'inverse', "sum"]:
            raise ValueError("normalize must be None, 'inverse', or 'sum'.")
        self.normalize = normalize

        RNN = getattr(layers, recurrent_model)

        self.recurrent_layers = []
        for i, units in enumerate(self.recurrent_layer_sizes):
            layer_name = f"{self.name}_rnn_{i}"
            cell_name = f"{layer_name}_cell"
            if self.direction == 'bidirectional':
                cell = RNN(units=units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout,
                           return_sequences=True, name=cell_name)
                rnn_layer = layers.Bidirectional(cell, name=layer_name)
            elif self.direction == 'forward':
                rnn_layer = RNN(units=units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout,
                                return_sequences=True, name=layer_name)
            elif self.direction == 'backward':
                rnn_layer = RNN(units=units, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout,
                                return_sequences=True, go_backwards=True, name=layer_name)
            else:
                raise ValueError("direction must be 'bidirectional', 'forward', or 'backward'.")
            self.recurrent_layers.append(rnn_layer)

        self.final_deep_dense = DeepLayer(final_hidden_layer_sizes+[bottleneck], 
                                     activation=final_hidden_activation,
                                     last_activation=final_activation,
                                     dropout_rate=dropout,
                                     name=f"{self.name}_finaldeep")       

    def build(self, input_shape):
        for layer in self.recurrent_layers:
            layer.build(input_shape)
            input_shape = layer.compute_output_shape(input_shape)
        self.final_deep_dense.build(input_shape)
        super(DeepRecurrentLayer, self).build(input_shape)

    def call(self, inputs):
        x = inputs
        for layer in self.recurrent_layers:
            x = layer(x)
        outputs = self.final_deep_dense(x)
        if self.normalize is not None:
            outputs = CustomNormalizationLayer(self.normalize, axis=-2, name=f"{self.name}_norm")(outputs)
        if outputs.shape[-1] == 1:
            return tf.squeeze(outputs, axis=-1)
        return outputs
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'recurrent_layer_sizes':      self.recurrent_layer_sizes,
            'final_activation':           self.final_activation,
            'final_hidden_layer_sizes':   self.final_hidden_layer_sizes,
            'final_hidden_activation':    self.final_hidden_activation,
            'direction':                  self.direction,
            'dropout':                    self.dropout,
            'recurrent_dropout':          self.recurrent_dropout,
            'recurrent_model':            self.recurrent_model,
            'normalize':                  self.normalize,
            'bottleneck':                 self.bottleneck
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable(package='Custom_Layers', name='TakeTop')
class TakeTop(layers.Layer):
    def __init__(self, **kwargs):
        super(TakeTop, self).__init__(**kwargs)
    
    def call(self, inputs):
        Mat, target_Mat = inputs
        
        target_shape = tf.shape(target_Mat)
        target_M = target_shape[1]
        return Mat[:, :target_M]
    def compute_output_shape(self, input_shape):
        Mat_shape, target_Mat_shape = input_shape
        return (Mat_shape[0], target_Mat_shape[1])
    def get_config(self):
        config = super(TakeTop, self).get_config()
        return config

@tf.keras.utils.register_keras_serializable(package='CCC_Functions', name='SVDReconstructFromFullLayer')
class SVDReconstructFromFullLayer(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
    
    def call(self, inputs):
        s_k, U_full, V_full = inputs
        return hf.svd_reconstruct_from_full(s_k, U_full, V_full)
    
    def get_config(self):
        return super().get_config()
