# 

[![PyPI version](https://img.shields.io/pypi/v/compact-rienet.svg)](https://pypi.org/project/compact-rienet/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**This library implements the neural estimators introduced in:**
- **Manolakis, E., Bongiorno, C., & Mantegna, R. N. (2025). Physics-Informed Singular-Value Learning for Cross-Covariances Forecasting in Financial Markets. Working Paper.**


## Key Features

- **Physics-Informed Neural Architecture**: The model utilizes a custom neural architecture designed to operate directly in the empirical singular-vector basis. By parameterizing the cleaning process as a nonlinear map from empirical singular values to cleaned values, the network explicitly enforces the problem's symmetries and rotational invariance implied by Random Matrix Theory (RMT).

- **Random Matrix Theory (RMT) as a Limiting Case**: The architecture is constructed to embed the asymptotically optimal Benaych-Georges, Bouchaud, and Potters (BBP) analytical solution as a special case. This "physics-informed" constraint ensures the model can recover theoretically optimal denoising in stationary regimes while providing the flexibility to adapt to real-world non-stationary dynamics and macroscopic market modes.

- **RIE-Style Cross-Covariance Cleaning**: Building on Rotationally Invariant Estimators (RIE), the method preserves the empirical singular vectors while performing nonlinear shrinkage on the singular values. This approach generalizes classical covariance cleaning to the rectangular cross-covariance setting, effectively characterizing comovements between different sets of assets even in high-dimensional regimes.

- **Robust End-to-End Forecasting**: Unlike purely analytical cleaners that rely on strict stationarity and bounded spectra, this framework is trained end-to-end to minimize out-of-sample (OOS) reconstruction error. The design is dimension-agnostic, allowing a model trained on one range of assets to be deployed across different universe sizes and relative dimensions without retraining.

## Installation
Install from source:

```bash
git clone https://github.com/Efstratios7/CrossRIE.git
cd CrossRIE
pip install -e .
```

## Quick Start

### Basic Usage
The core component is the `CrossRIELayer`. It expects four inputs: the two marginal covariance matrices \newline ($\mathbf{C}_{XX}$, $\mathbf{C}_{YY}$), the cross-correlation matrix ($\mathbf{C}_{XY}$), and the number of samples ($n$).

```python
import tensorflow as tf
from crossrie.layer import CrossRIELayer

# Initialize the layer
# By default, it returns the cleaned Cross-Correlation matrix 'Cxy'
cross_rie = CrossRIELayer(
    encoding_units=[16, 2],
    lstm_units=[128, 64],
    outputs=['Cxy', 'Sxy']
)

# Generate dummy data (Batch, N, M)
B, N, M, T = 32, 10, 12, 100
Cxx = tf.random.normal((B, N, N))
Cyy = tf.random.normal((B, M, M))
Cxy = tf.random.normal((B, N, M))
n_samples = tf.constant([T] * B) # Number of samples which are used to compute the covariance matrices Cxx and Cyy

# Forward pass
outputs = cross_rie([Cxx, Cyy, Cxy, n_samples])

Cxy_clean = outputs['Cxy']      # Denoised Cross-Correlation
Sxy_clean = outputs['Sxy']      # Cleaned Singular Values

print("Cleaned Cxy shape:", Cxy_clean.shape)
print("Cleaned Sxy shape:", Sxy_clean.shape)
```

### Training
The layer is fully differentiable and can be trained using standard Keras optimization workflows.

```python
import tensorflow as tf
from keras import Model, Input
from crossrie.layer import CrossRIELayer

B, N, M, T = 32, 10, 12, 100

def create_model():
    # Shapes are (None, None) to allow variable sequence lengths
    input_cxx = Input(shape=(None, None), name='Cxx')
    input_cyy = Input(shape=(None, None), name='Cyy')
    input_cxy = Input(shape=(None, None), name='Cxy')
    input_n = Input(shape=(1,), name='n_samples')
    
    # Forward pass
    cxy_clean = CrossRIELayer(outputs=['Cxy'])([input_cxx, input_cyy, input_cxy, input_n])
    
    return Model(inputs=[input_cxx, input_cyy, input_cxy, input_n], outputs=cxy_clean)

model = create_model()
model.compile(optimizer='adam', loss='mse')

# Training Data
# In a real scenario, these would be computed from your data.
# Cxx: (Batch, N, N), Cyy: (Batch, M, M), Cxy: (Batch, N, M)
Cxx = tf.random.normal((B, N, N))
Cyy = tf.random.normal((B, M, M))
Cxy = tf.random.normal((B, N, M))

# n_samples must match the batch dimension. 
# It represents the number of time steps T used to compute the correlations/covariances.
# Shape: (Batch, 1) or (Batch,)
n_samples = tf.constant([[float(T)] for _ in range(B)]) 

# Target Variable (Cleaned Cxy)
# In supervised learning, this would be the "true" cross-correlation.
Y_target = tf.random.normal((B, N, M))

model.fit([Cxx, Cyy, Cxy, n_samples], Y_target, epochs=1, batch_size=32)
```

### Different Output Types
You can configure the layer to return different components by passing a list of keys to `outputs`.

- `'Cxy'`: The reconstructed, denoised cross-correlation matrix.
- `'Sxy'`: The vector of cleaned singular values.

```python
# Returns only the cleaned singular values
layer_s = CrossRIELayer(outputs=['Sxy']) # When creating the model use Sxy to receive the cleaned singular values instead the clean cross-correlation matrix Cxy
s_tilde = layer_s([Cxx, Cyy, Cxy, n_samples])
```

## Requirements
- Python >= 3.8
- TensorFlow >= 2.10.0
- Keras >=3.12.0
- NumPy >= 1.26.4

## Development
```bash
git clone https://github.com/Efstratios7/CrossRIE.git
cd CrossRIE
pip install -e ".[dev]"
pytest tests/
```

## Citation

## Support
For questions, issues, or contributions, please:

- Open an issue on [GitHub](https://github.com/bongiornoc/Compact-RIEnet/issues)
- Check the documentation
- Contact Efstratios Manolakis (<stratomanolaki@gmail.com>)
- Contact Prof. Christian Bongiorno (<christian.bongiorno@centralesupelec.fr>) for calibrated model weights or collaboration requests
