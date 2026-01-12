import tensorflow as tf
from keras import layers
from keras import backend as K
from typing import Optional, List, Tuple, Union, Any

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@tf.function(reduce_retracing=True)
def _symmetrize(M: tf.Tensor) -> tf.Tensor:
    """
    Symmetrizes a matrix M: 0.5 * (M + M^T).
    
    Args:
        M: Input tensor [..., N, N].
        
    Returns:
        Symmetrized tensor [..., N, N].
    """
    return 0.5 * (M + tf.linalg.matrix_transpose(M))

@tf.function(reduce_retracing=True)
def svd_via_eigh_full(C: tf.Tensor, 
                      eps: Optional[float] = None, 
                      jitter_eigh: float = 0.0) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
   """
   Batched SVD via eigh(CC^T) con V_full costruita in modo coerente con U_k e s_k.
   C: [B, n, m]
   Ritorna:
     s_k    [B, r]        (r = min(n,m), singolari in ordine decrescente)
     U_k    [B, n, r]
     V_full [B, m, m]     (prime r colonne = right singular vectors coerenti)
   """
   C = tf.convert_to_tensor(C)
   if eps is None:
       eps = tf.cast(K.epsilon(), C.dtype)

   B = tf.shape(C)[0]
   n = tf.shape(C)[1]
   m = tf.shape(C)[2]
   r = tf.minimum(n, m)
   m_minus_r = m - r

   # 1) U_k e s_k da eigh(CC^T), simmetrizzato
   A = tf.matmul(C, C, transpose_b=True)            # [B, n, n]
   A = _symmetrize(A)

   if jitter_eigh != 0.0:
       I_n = tf.eye(n, batch_shape=[B], dtype=C.dtype)
       A = A + tf.cast(jitter_eigh, C.dtype) * I_n  # piccolo jitter sulla diagonale

   evals_u, U_full = tf.linalg.eigh(A)              # autovalori in ordine crescente

   # Ordina in ordine decrescente
   idx_u = tf.argsort(evals_u, direction="DESCENDING")
   evals_u = tf.gather(evals_u, idx_u, batch_dims=1, axis=1)    # [B, n]
   U_full = tf.gather(U_full, idx_u, batch_dims=1, axis=2)      # [B, n, n]

   # Singolari = sqrt(max(evals, 0))
   zeros_evals = tf.zeros_like(evals_u)
   s_all = tf.sqrt(tf.maximum(evals_u, zeros_evals))            # [B, n]

   U_k = U_full[:, :, :r]                                       # [B, n, r]
   s_k = s_all[:, :r]                                           # [B, r]
   s_safe = tf.maximum(s_k, eps)

   # 2) prime r colonne di V: V1 = C^T U_k / s_k
   V1 = tf.matmul(C, U_k, transpose_a=True)                     # [B, m, r]
   V1 = V1 / tf.expand_dims(s_safe, axis=1)                     # [B, m, r]

   # 3) normalizza colonne di V1 e compensa in s_k
   norms = tf.maximum(tf.linalg.norm(V1, axis=1), eps)          # [B, r]
   V1 = V1 / tf.expand_dims(norms, axis=1)                      # [B, m, r]
   s_k = s_k * norms                                            # [B, r]

   I_m = tf.eye(m, batch_shape=[B], dtype=C.dtype)              # [B, m, m]
   W0 = I_m[:, :, r:]                                           # [B, m, m-r]

   # Proietta W0 sul complemento di span(V1):
   # V1^T W0: [B, r, m-r]
   V1tW0 = tf.matmul(V1, W0, transpose_a=True)
   # W1 = W0 - V1 (V1^T W0): [B, m, m-r]
   W1 = W0 - tf.matmul(V1, V1tW0)

   # QR di W1: le colonne di Q sono ortonormali e (idealmente) nel complemento di span(V1)
   # Funziona anche quando m_minus_r == 0 (dimensione nulla).
   V2, _ = tf.linalg.qr(W1, full_matrices=False)                # [B, m, m-r]

   # V_full = [V1 | V2]
   V_full = tf.concat([V1, V2], axis=2)                         # [B, m, m]

   return s_k, U_full, V_full

@tf.function(reduce_retracing=True)
def diag_vtCv(C: tf.Tensor, V: tf.Tensor) -> tf.Tensor:
    """
    Computes diagonal elements of V^T C V.
    
    Args:
        C: Covariance/Correlation matrix [B, n, n].
        V: Matrix of vectors (e.g., eigenvectors) [B, n, r].
        
    Returns:
        Diagonal elements [B, r].
    """
    CV = tf.matmul(C, V)           # [B, n, r]
    d  = tf.reduce_sum(V * CV, axis=1)  # somma su dimensione n -> [B, r]
    return d

def pad_A_to_B_simple(A: tf.Tensor, B: tf.Tensor) -> tf.Tensor:
    """
    Pads tensor A to match dimensions of B (or max dimension of B).
    
    Args:
        A: Input tensor to pad.
        B: Reference tensor for dimensions.
        
    Returns:
        Padded tensor.
    """
    n_a = tf.shape(A)[1]
    n_b = tf.maximum(tf.shape(B)[1], tf.shape(B)[2])
    
    pad_len = n_b - n_a
    paddings = tf.stack([[0, 0], [0, pad_len], [0, 0]], axis=0)
    return tf.pad(A, paddings, mode="CONSTANT", constant_values=0)

@tf.function(reduce_retracing=True)
def svd_reconstruct_from_full(s_k: tf.Tensor, 
                              U_full: tf.Tensor, 
                              V_full: tf.Tensor) -> tf.Tensor:
    """
    Ricostruisce C_hat = U_k @ diag(s_k) @ (V_full[:,:,:r])^T
    """
    r = tf.shape(s_k)[1]
    V_k = V_full[:, :, :r]   
    U_k = U_full[:, :, :r]   # [B, m, r]
    S = tf.linalg.diag(s_k)                             # [B, r, r]
    return tf.matmul(U_k, tf.matmul(S, V_k, transpose_b=True))

# ============================================================================
# CUSTOM LAYERS
# ============================================================================

@tf.keras.utils.register_keras_serializable(package='Custom_Layers', name='ZScoreLayer')
class ZScoreLayer(layers.Layer):
    """
    Layer that performs Z-Score normalization on the last axis.
    """
    def __init__(self, 
                 epsilon: float = 1e-6, 
                 name: Optional[str] = None, 
                 **kwargs):
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
    """
    Layer that computes the length of the time series (last dimension).
    """
    def __init__(self, 
                 dtype: tf.DType = tf.float32, 
                 name: Optional[str] = None, 
                 **kwargs):
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
    """
    Layer that ensures a minimum sample count value.
    """
    def __init__(self, 
                 minimum: float = 1.0, 
                 name: Optional[str] = None, 
                 **kwargs):
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
    """
    Layer that computes the Covariance matrix X * X^T (optionally normalized).
    """
    def __init__(self, 
                 expand_dims: bool = False, 
                 normalize: bool = True, 
                 name: Optional[str] = None, 
                 **kwargs):
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
    """
    Layer that computes Cross-Covariance matrix (X * Y^T) / denom.
    """
    def __init__(self, 
                 name: Optional[str] = None, 
                 **kwargs):
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
    """
    Layer to expand dimensions of input tensor.
    """
    def __init__(self, 
                 axis: int = -1, 
                 **kwargs):
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
    """
    Layer wrapper for SVD via Eigh.
    """
    def __init__(self, 
                 eps: Optional[float] = None, 
                 name: Optional[str] = None, 
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.eps = eps
    
    def call(self, C):
        return svd_via_eigh_full(C, eps=self.eps)
    
    def get_config(self):
        config = super().get_config()
        config.update({'eps': self.eps})
        return config

@tf.keras.utils.register_keras_serializable(package='CCC_Functions', name='DiagVtCvLayer')
class DiagVtCvLayer(tf.keras.layers.Layer):
    """
    Layer wrapper for computing diagonal of V^T C V.
    """
    def __init__(self, 
                 name: Optional[str] = None, 
                 **kwargs):
        super().__init__(name=name, **kwargs)
    
    def call(self, inputs):
        C, V = inputs
        return diag_vtCv(C, V)
    
    def get_config(self):
        return super().get_config()

@tf.keras.utils.register_keras_serializable(package='CCC_Functions', name='DimAware2DLayer')
class DimAware2DLayer(tf.keras.layers.Layer):
    """
    Adds dimension-aware features (like N/T, M/T) to the input tensor.
    """
    def __init__(self, 
                 features: List[str] = ['n1', 'n2', 'q1', 'q2'], 
                 name: Optional[str] = None, 
                 **kwargs):
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
    """
    Layer to pad input A to match shape of B.
    """
    def __init__(self, 
                 name: Optional[str] = None, 
                 **kwargs):
        super().__init__(name=name, **kwargs)
    
    def call(self, inputs):
        A, B = inputs
        return pad_A_to_B_simple(A, B)
    
    def get_config(self):
        return super().get_config()

@tf.keras.utils.register_keras_serializable(package='Custom_Layers', name='DeepLayer')
class DeepLayer(layers.Layer):
    """
    A deep fully connected network with dropouts and optional biases.
    """
    def __init__(self, 
                 hidden_layer_sizes: List[int], 
                 last_activation: str = "linear",
                 activation: str = "leaky_relu", 
                 other_biases: bool = True, 
                 last_bias: bool = True,
                 dropout_rate: float = 0., 
                 kernel_initializer: str = "glorot_uniform", 
                 name: Optional[str] = None, 
                 **kwargs):
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
    """
    Performs Sum or Inverse normalization along an axis.
    """
    def __init__(self, 
                 mode: str = 'sum', 
                 axis: int = -2, 
                 name: Optional[str] = None, 
                 **kwargs):
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
    """
    A deep recurrent network followed by a deep fully connected network.
    """
    def __init__(self, 
                 recurrent_layer_sizes: List[int],
                 final_activation: str = "softplus", 
                 final_hidden_layer_sizes: List[int] = [], 
                 final_hidden_activation: str = "leaky_relu",
                 direction: str = 'bidirectional', 
                 dropout: float = 0.,
                 recurrent_dropout: float = 0.,
                 recurrent_model: str = 'LSTM', 
                 normalize: Optional[str] = None, 
                 bottleneck: int = 1, 
                 name: Optional[str] = None, 
                 **kwargs):
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
    """
    Layer that slices the first M columns of the input matrix.
    """
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
    """
    Reconstructs matrix from SVD components.
    """
    def __init__(self, 
                 name: Optional[str] = None, 
                 **kwargs):
        super().__init__(name=name, **kwargs)
    
    def call(self, inputs):
        s_k, U_full, V_full = inputs
        return svd_reconstruct_from_full(s_k, U_full, V_full)
    
    def get_config(self):
        return super().get_config()
