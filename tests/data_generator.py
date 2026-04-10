import tensorflow as tf
import numpy as np

def dynamic_matrix_generator(batch_size=16, N_range=(20, 50), M_range=(20, 50),ndays_range=(200,600)):
    """Generates a batch of correctly shaped, dynamic data tensors."""
    while True:
        # Variable Dimensions
        N = np.random.randint(N_range[0], N_range[1]+1)
        M = np.random.randint(M_range[0], M_range[1]+1)
        ndays = np.random.randint(ndays_range[0], ndays_range[1]+1)
        
        # PSD matrices for Cxx and Cyy
        X_N = tf.random.normal((batch_size, N, ndays))
        Cxx = tf.matmul(X_N, X_N, transpose_b=True)
        
        X_M = tf.random.normal((batch_size, M, ndays))
        Cyy = tf.matmul(X_M, X_M, transpose_b=True)


        Cxy_clean = tf.matmul(X_N, X_M, transpose_b=True) / ndays
        
        # Dense input cross-correlations
        Cxy_noisy = Cxy_clean +tf.random.normal(mean=0,stddev=0.01,shape=(batch_size, N, M))
        T_samples = tf.constant([float(ndays)] * batch_size)
        
        yield (Cxx, Cyy, Cxy_noisy, T_samples), Cxy_clean

def get_dynamic_dataset(batch_size=16, N_range=(20, 50), M_range=(20, 50),ndays_range=(200,600)):
    """Builds a tensorflow dataset out of the dynamic data pipeline."""
    return tf.data.Dataset.from_generator(
        lambda: dynamic_matrix_generator(batch_size, N_range, M_range,ndays_range),
        output_signature=(
            (
                tf.TensorSpec(shape=(None, None, None), dtype=tf.float32), # Cxx
                tf.TensorSpec(shape=(None, None, None), dtype=tf.float32), # Cyy
                tf.TensorSpec(shape=(None, None, None), dtype=tf.float32), # Cxy_noisy
                tf.TensorSpec(shape=(None,), dtype=tf.float32),            # T_samples
            ),
            tf.TensorSpec(shape=(None, None, None), dtype=tf.float32)      # Cxy_clean
        )
    )
