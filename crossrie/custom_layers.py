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
def spectral_svd_decomposition(C: tf.Tensor, 
                               eps: Optional[float] = None, 
                               jitter_eigh: float = 0.0) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
   """
   Computes a batched SVD decomposition using `tf.linalg.eigh` on the covariance matrix C * C^T.
   
   Args:
       C: Input tensor of shape [B, N, M].
       eps: Epsilon for numerical stability. If None, uses backend epsilon.
       jitter_eigh: Small jitter added to the diagonal for stability during eigendecomposition.
       
   Returns:
       s_k: Singular values of shape [B, r], where r = min(N, M). Sorted in descending order.
       U_full: Left singular vectors of shape [B, N, N].
       V_full: Right singular vectors of shape [B, M, M], where the first r columns are consistent with s_k and U_full.
   """
   C = tf.convert_to_tensor(C)
   if eps is None:
       eps = tf.cast(K.epsilon(), C.dtype)

   B = tf.shape(C)[0]
   n = tf.shape(C)[1]
   m = tf.shape(C)[2]
   r = tf.minimum(n, m)
   m_minus_r = m - r

   # 1) Compute U_k and s_k from eigh(C * C^T)
   A = tf.matmul(C, C, transpose_b=True)            # [B, n, n]
   A = _symmetrize(A)

   if jitter_eigh != 0.0:
       I_n = tf.eye(n, batch_shape=[B], dtype=C.dtype)
       A = A + tf.cast(jitter_eigh, C.dtype) * I_n  

   evals_u, U_full = tf.linalg.eigh(A)              

   # Sort in descending order
   idx_u = tf.argsort(evals_u, direction="DESCENDING")
   evals_u = tf.gather(evals_u, idx_u, batch_dims=1, axis=1)    # [B, n]
   U_full = tf.gather(U_full, idx_u, batch_dims=1, axis=2)      # [B, n, n]

   # Singular values = sqrt(max(evals, 0))
   zeros_evals = tf.zeros_like(evals_u)
   s_all = tf.sqrt(tf.maximum(evals_u, zeros_evals))            # [B, n]

   U_k = U_full[:, :, :r]                                       # [B, n, r]
   s_k = s_all[:, :r]                                           # [B, r]
   s_safe = tf.maximum(s_k, eps)

   # 2) Compute first r columns of V: V1 = C^T U_k / s_k
   V1 = tf.matmul(C, U_k, transpose_a=True)                     # [B, m, r]
   V1 = V1 / tf.expand_dims(s_safe, axis=1)                     # [B, m, r]

   # 3) Normalize V1 columns and compensate in s_k
   norms = tf.maximum(tf.linalg.norm(V1, axis=1), eps)          # [B, r]
   V1 = V1 / tf.expand_dims(norms, axis=1)                      # [B, m, r]
   s_k = s_k * norms                                            # [B, r]

   I_m = tf.eye(m, batch_shape=[B], dtype=C.dtype)              # [B, m, m]
   W0 = I_m[:, :, r:]                                           # [B, m, m-r]

   # Project W0 onto the complement of span(V1)
   V1tW0 = tf.matmul(V1, W0, transpose_a=True)
   W1 = W0 - tf.matmul(V1, V1tW0)

   # QR decomposition of W1
   V2, _ = tf.linalg.qr(W1, full_matrices=False)                # [B, m, m-r]

   # V_full = [V1 | V2]
   V_full = tf.concat([V1, V2], axis=2)                         # [B, m, m]

   return s_k, U_full, V_full

@tf.function(reduce_retracing=True)
def compute_projected_variance_diagonal(C: tf.Tensor, V: tf.Tensor) -> tf.Tensor:
    """
    Computes diagonal elements of the projected covariance term V^T C V.
    
    Args:
        C: Covariance/Correlation matrix [B, N, N].
        V: Projection matrix (e.g., eigenvectors) [B, N, r].
        
    Returns:
        Diagonal elements [B, r].
    """
    CV = tf.matmul(C, V)           # [B, n, r]
    d  = tf.reduce_sum(V * CV, axis=1)  # sum over dimension n -> [B, r]
    return d

def match_dimensions_by_padding(A: tf.Tensor, B: tf.Tensor) -> tf.Tensor:
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
def reconstruct_matrix_from_svd(s_k: tf.Tensor, 
                                U_full: tf.Tensor, 
                                V_full: tf.Tensor) -> tf.Tensor:
    """
    Reconstructs the rectangular matrix from full component set.

    Args:
        s_k: Singular values [B, r].
        U_full: Left singular vectors [B, N, N].
        V_full: Right singular vectors [B, M, M].

    Returns:
        Reconstructed matrix [B, N, M].
    """
    r = tf.shape(s_k)[1]
    V_k = V_full[:, :, :r]   
    U_k = U_full[:, :, :r]   # [B, m, r]
    S = tf.linalg.diag(s_k)                             # [B, r, r]
    return tf.matmul(U_k, tf.matmul(S, V_k, transpose_b=True))

# ============================================================================
# CUSTOM LAYERS
# ============================================================================


@tf.keras.utils.register_keras_serializable(package='crossrie', name='ExpandDimsLayer')
class ExpandDimsLayer(layers.Layer):
    """
    Layer to expand dimensions of input tensor.

    Args:
        axis: Axis index to expand.
        **kwargs: Standard Keras Layer arguments.

    Returns:
        Tensor with expanded dimensions.
    """
    def __init__(self, 
                 axis: int = -1, 
                 **kwargs):
        super(ExpandDimsLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        """
        Expands intended dimension.

        Args:
            inputs: Input tensor. For Sxy: [Batch, K].

        Returns:
            Expanded tensor. [Batch, K, 1].
        """
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

@tf.keras.utils.register_keras_serializable(package='crossrie', name='SpectralSVDLayer')
class SpectralSVDLayer(tf.keras.layers.Layer):
    """
    Layer wrapper for Spectral SVD Decomposition.

    Performs batched SVD on the input tensor C * C^T.

    Args:
        eps: Epsilon for numerical stability.
        name: Name of the layer.
        **kwargs: Standard Keras Layer arguments.

    Returns:
        Tuple (s_k, U_full, V_full) containing singular values, left and right singular vectors.
    """
    def __init__(self, 
                 eps: Optional[float] = None, 
                 name: Optional[str] = None, 
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.eps = eps
    
    def call(self, C):
        """
        Performs SVD.

        Args:
            C: Input matrix [Batch, N, M].

        Returns:
            Tuple (s_k, U_full, V_full):
            - s_k: Singular values [Batch, K], where K=min(N, M).
            - U_full: Left singular vectors [Batch, N, N].
            - V_full: Right singular vectors [Batch, M, M].
        """
        return spectral_svd_decomposition(C, eps=self.eps)
    
    def get_config(self):
        config = super().get_config()
        config.update({'eps': self.eps})
        return config

@tf.keras.utils.register_keras_serializable(package='crossrie', name='ProjectedVarianceDiagonalLayer')
class ProjectedVarianceDiagonalLayer(tf.keras.layers.Layer):
    """
    Layer wrapper for computing diagonal of V^T C V.

    Args:
        name: Name of the layer.
        **kwargs: Standard Keras Layer arguments.

    Returns:
        Diagonal elements of the projected variance.
    """
    def __init__(self, 
                 name: Optional[str] = None, 
                 **kwargs):
        super().__init__(name=name, **kwargs)
    
    def call(self, inputs):
        """
        Computes projected variance diagonal.

        Args:
            inputs: Tuple (C, V).
            C: Block matrix [Batch, N, N] (for Cxx) or [Batch, M, M] (for Cyy).
            V: Corresponding singular vector matrix [Batch, N, N] (Lxy) or [Batch, M, M] (Rxy).

        Returns:
            Diagonal elements. [Batch, N] (for Cxx) or [Batch, M] (for Cyy).
        """
        C, V = inputs
        return compute_projected_variance_diagonal(C, V)
    
    def get_config(self):
        return super().get_config()

@tf.keras.utils.register_keras_serializable(package='crossrie', name='DimensionAwarenessLayer')
class DimensionAwarenessLayer(tf.keras.layers.Layer):
    """
    Adds dimension-aware features (like N/T, M/T) to the input tensor.

    Concatenates additional features representing system dimensions and ratios to the input.

    Args:
        features: List of feature names to add ('n1', 'n2', 'q1', 'q2', 't', 't1', 't2').
        name: Name of the layer.
        **kwargs: Standard Keras Layer arguments.

    Returns:
        Tensor with concatenated dimension features.
    """
    def __init__(self, 
                 features: List[str] = ['n1', 'n2', 'q1', 'q2'], 
                 name: Optional[str] = None, 
                 **kwargs):
        super().__init__(name=name, **kwargs)
        valid_keys = {'n1', 'n2', 'q1', 'q2', 't', 't1', 't2'} 
        for f in features:
            if f not in valid_keys:
                raise ValueError(f"Feature '{f}' not supported. Use: {valid_keys}")
        self.features = features

    def compute_output_shape(self, input_shape):
        if len(input_shape) < 2:
             raise ValueError("DimensionAwarenessLayer requires [Mat, Shape_mat, t]")

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
            raise ValueError(f"Unexpected Mat shape: {mat_shape}. Expected (Batch, N) or (Batch, N, C)")

    def call(self, inputs):
        """
        Adds dimension features.

        Args:
            inputs: Tuple (Mat, Shape_mat, t).
            Mat: Input matrix (Pxx/Pyy) [Batch, N, 1] (or [Batch, M, 1]).
            Shape_mat: Input shape matrix (Cxy) [Batch, N, M].
            t: Input time vector (T_samples) [Batch].

        Returns:
            Tensor with added features. [Batch, N, 2] (or [Batch, M, 2]).
        """
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

@tf.keras.utils.register_keras_serializable(package='crossrie', name='DimensionMatchingLayer')
class DimensionMatchingLayer(tf.keras.layers.Layer):
    """
    Layer to pad input A to match shape of B.

    Computes necessary padding based on dimensions of B and applies it to A.

    Args:
        name: Name of the layer.
        **kwargs: Standard Keras Layer arguments.

    Returns:
        Padded tensor A matching dimensions of B.
    """
    def __init__(self, 
                 name: Optional[str] = None, 
                 **kwargs):
        super().__init__(name=name, **kwargs)
    
    def call(self, inputs):
        """
        Pads tensor A.

        Args:
            inputs: Tuple (A, B).
            A: Input tensor to pad [Batch, N, F] (or [Batch, M, F]).
            B: Reference tensor (Cxy) [Batch, N, M].

        Returns:
            Padded A. [Batch, Max(N,M), F].
        """
        A, B = inputs
        return match_dimensions_by_padding(A, B)
    
    def get_config(self):
        return super().get_config()

@tf.keras.utils.register_keras_serializable(package='crossrie', name='DeepLayer')
class DeepLayer(layers.Layer):
    """
    A deep fully connected network with dropouts and optional biases.

    Args:
        hidden_layer_sizes: List of integers specifying the size of each hidden layer.
        last_activation: Activation function for the output layer.
        activation: Activation function for hidden layers.
        other_biases: Whether to use bias in hidden layers.
        last_bias: Whether to use bias in the output layer.
        dropout_rate: Dropout rate applied after each hidden layer.
        kernel_initializer: Initializer for weight matrices.
        name: Name of the layer.
        **kwargs: Standard Keras Layer arguments.

    Returns:
        Output tensor after passing through the deep network.
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
        """
        Forward pass.

        Args:
            inputs: Input tensor [Batch, Max(N,M), Features].

        Returns:
            Output tensor [Batch, Max(N,M), Hidden] (if deep layer inside DeepRecurrent) or [Batch, Max(N,M)] (after squeeze).
        """
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

@tf.keras.utils.register_keras_serializable(package='crossrie', name='CustomNormalizationLayer')
class CustomNormalizationLayer(layers.Layer):
    """
    Performs Sum or Inverse normalization along an axis.

    Args:
        mode: 'sum' (normalize by sum) or 'inverse' (normalize by sum of inverses).
        axis: Axis along which to normalize.
        name: Name of the layer.
        **kwargs: Standard Keras Layer arguments.

    Returns:
        Normalized tensor.
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
        """
        Normalizes input.

        Args:
            x: Input tensor.

        Returns:
            Normalized tensor.
        """
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

@tf.keras.utils.register_keras_serializable(package='crossrie', name='DeepRecurrentLayer')
class DeepRecurrentLayer(layers.Layer):
    """
    A deep recurrent network followed by a deep fully connected network.

    Args:
        recurrent_layer_sizes: List of integers, sizes of recurrent layers.
        final_activation: Activation function for the final output.
        final_hidden_layer_sizes: Sizes of hidden layers in the final dense network.
        final_hidden_activation: Activation for hidden layers in the final dense network.
        direction: 'bidirectional', 'forward', or 'backward'.
        dropout: Dropout rate for dense layers.
        recurrent_dropout: Dropout rate for recurrent layers.
        recurrent_model: 'LSTM' or 'GRU'.
        normalize: Normalization mode ('sum', 'inverse', or None).
        bottleneck: Size of the bottleneck layer.
        name: Name of the layer.
        **kwargs: Standard Keras Layer arguments.

    Returns:
        Output tensor from the deep recurrent architecture.
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
        """
        Forward pass of Deep RNN.

        Args:
            inputs: Tokenized input tensor [Batch, Max(N,M), Features].

        Returns:
            Output tensor [Batch, Max(N,M)].
        """
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

    Args:
        **kwargs: Standard Keras Layer arguments.

    Returns:
        Tensor sliced to match the second dimension of the target matrix.
    """
    def __init__(self, **kwargs):
        super(TakeTop, self).__init__(**kwargs)

    
    def call(self, inputs):
        """
        Slices input matrix.

        Args:
            inputs: Tuple (Mat, target_Mat).
            Mat: Input matrix (Shrinkage) [Batch, Max(N,M)].
            target_Mat: Target matrix (Sxy) [Batch, K] where K=min(N, M).

        Returns:
            Sliced matrix [Batch, K].
        """
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

@tf.keras.utils.register_keras_serializable(package='crossrie', name='SVDReconstructionLayer')
class SVDReconstructionLayer(tf.keras.layers.Layer):
    """
    Reconstructs matrix from SVD components.

    Computes C_hat = U_k @ diag(s_k) @ V_full[:, :, :r]^T.

    Args:
        name: Name of the layer.
        **kwargs: Standard Keras Layer arguments.

    Returns:
        Reconstructed matrix tensor.
    """
    def __init__(self, 
                 name: Optional[str] = None, 
                 **kwargs):
        super().__init__(name=name, **kwargs)
    
    def call(self, inputs):
        """
        Reconstructs matrix.

        Args:
            inputs: Tuple (s_k, U_full, V_full).
            s_k: Singular values (Sxy_cleaned) [Batch, K].
            U_full: Left singular vectors (Lxy) [Batch, N, N].
            V_full: Right singular vectors (Rxy) [Batch, M, M].

        Returns:
            Reconstructed matrix (Cxy_denoised) [Batch, N, M].
        """
        s_k, U_full, V_full = inputs
        return reconstruct_matrix_from_svd(s_k, U_full, V_full)
    
    def get_config(self):
        return super().get_config()

@tf.keras.utils.register_keras_serializable(package='crossrie', name='Two_Stream_EncoderLayer')
class Two_Stream_EncoderLayer(layers.Layer):
    """
    Two-stream encoder layer that processes paired inputs and aggregates them.

    Args:
        encoding_units: List of integers for the shared encoder (MLP) structure.
        lstm_units: List of integers for the recurrent aggregator hidden sizes.
        final_hidden_layer_sizes: List of integers for the final hidden layers.
        final_activation: Activation function for the final output.
        name: Name of the layer.
        **kwargs: Standard Keras Layer arguments.

    Returns:
        Aggregated head tensor.
    """
    def __init__(self,
                 encoding_units: List[int] = [16, 2],
                 lstm_units: List[int] = [128, 64],
                 final_hidden_layer_sizes: List[int] = [252],
                 final_activation: str = 'linear',
                 name: Optional[str] = None,
                 **kwargs):
        super(Two_Stream_EncoderLayer, self).__init__(name=name, **kwargs)
        self.encoding_units = encoding_units
        self.lstm_units = lstm_units
        self.final_hidden_layer_sizes = final_hidden_layer_sizes
        self.final_activation = final_activation

        if self.encoding_units:
            self.encoder = DeepLayer(hidden_layer_sizes=self.encoding_units, name='Encoder_Deep')
        else:
            self.encoder = None

        self.shrinkage = DeepRecurrentLayer(
            recurrent_layer_sizes=self.lstm_units,
            final_hidden_layer_sizes=self.final_hidden_layer_sizes,
            final_activation=self.final_activation,
            name='DeepRecurrent'
        )

    def build(self, input_shape):
        """
        Builds the layer.

        Args:
            input_shape: List of two input shapes [shape_Pxx_Sxy, shape_Pyy_Sxy] 
                         or a single shape if passed as a list.
                         Each shape is (Batch, M, Features).
        """
        # input_shape is likely a list of shapes [shape1, shape2]
        if isinstance(input_shape, list) and len(input_shape) > 0:
            first_shape = input_shape[0]
        else:
            first_shape = input_shape

        # first_shape is (None, None, Channels)
        total_channels = first_shape[-1]

        if self.encoder:
            self.encoder.build(first_shape)
            token_dim = self.encoding_units[-1]
        else:
            token_dim = total_channels
            
        self.shrinkage.build((first_shape[0], first_shape[1], token_dim))
        super(Two_Stream_EncoderLayer, self).build(input_shape)

    def call(self, inputs):
        """
        Forward pass.

        Args:
            inputs: List [Pxx_Sxy, Pyy_Sxy].
            Pxx_Sxy: Concatenated features for X stream [Batch, M, F].
            Pyy_Sxy: Concatenated features for Y stream [Batch, M, F].

        Returns:
            aggregator_head: Output from shrinkage layer.
        """
        Pxx_Sxy, Pyy_Sxy = inputs

        if self.encoder:
            Tokens = self.encoder(Pxx_Sxy) + self.encoder(Pyy_Sxy) 
        else:
            Tokens = Pxx_Sxy + Pyy_Sxy
            
        aggregator_head = self.shrinkage(Tokens)
        return aggregator_head

    def get_config(self):
        config = super(Two_Stream_EncoderLayer, self).get_config()
        config.update({
            'encoding_units': self.encoding_units,
            'lstm_units': self.lstm_units,
            'final_hidden_layer_sizes': self.final_hidden_layer_sizes,
            'final_activation': self.final_activation
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
