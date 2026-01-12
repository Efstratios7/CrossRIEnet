# Generalized CCC Model

## Paper
[Insert Paper Title and Link Here]

## Key Features
This repository hosts the **Generalized Cross-Correlation Correction (CCC) Model**, a deep learning-based framework for estimating and denoising high-dimensional cross-correlation matrices. Key capabilities include:

*   **Deep Non-Linear Shrinkage**: Utilizes Deep Neural Networks (DNN) and Recurrent Neural Networks (LSTM/GRU) to learn optimal non-linear shrinkage functions for singular values.
*   **Generalized Noise Modeling**: Supports both **Multiplicative** and **Additive** noise assumptions, allowing flexibility based on the underlying data characteristics.
*   **SVD-Based Architecture**: Implements an end-to-end differentiable Singular Value Decomposition (SVD) within the model (via `SVDViaEighFullLayer`), enabling direct optimization of spectral properties.
*   **Dimension Awareness**: Incorporates "Dimension Aware" layers that account for the ratio of features to samples ($N/T$ and $M/T$), making the model robust to varying system sizes.
*   **Flexible Configuration**: Customizable architecture including encoding sizes, recurrent unit types/sizes, and activation functions (e.g., Softplus for non-negative multiplicative shrinkage).

## Repository Structure
*   `ccc/`: Source package.
    *   `layers.py`: Contains the `GeneralizedCCCModel` definition.
    *   `custom_layers.py`: Custom Keras layers used by the model.
*   `tests/`: Unit tests.
*   `requirements.txt`: Python package dependencies.
*   `verification_script.py`: Script to verify imports and basic functionality.

## Installation

```bash
# Clone the repository
git clone [Insert Repository Link Here]
cd Cross-Covariance-Cleaning

# Create and activate the environment
conda env create -f environment.yml
conda activate ccc_env

# Install in editable mode
pip install -e .
```

## How to use it

The core model is defined in `ccc.layers` but can be imported directly from the top-level `ccc` package.

### Example

```python
import tensorflow as tf
from ccc import GeneralizedCCCModel

# 1. Initialize the model
#    - multiplicative=True ensures non-negative shrinkage (useful for variance/volatility)
#    - encoding_units: Dense layers before the LSTM
#    - lstm_units: Hidden state size of the recurrent shrinkage
model = GeneralizedCCCModel(
    encoding_units=[16, 8],
    lstm_units=[32],
    final_hidden_layer_sizes=[16],
    multiplicative=True,
    final_activation='softplus'
)

# 2. Prepare dummy data
#    B: Batch size
#    N, M: Dimensions of the two systems
B, N, M = 2, 50, 50
Cxx = tf.random.normal((B, N, N)) # Covariance/Correlation of System X
Cyy = tf.random.normal((B, M, M)) # Covariance/Correlation of System Y
Cxy = tf.random.normal((B, N, M)) # Cross-Correlation between X and Y
n_samples = tf.constant([100.0, 100.0]) # Number of effective samples per batch

# 3. Forward Pass
#    Returns the denoised Cross-Correlation matrix Cxy
denoised_Cxy = model([Cxx, Cyy, Cxy, n_samples])

print("Input shape:", Cxy.shape)
print("Denoised shape:", denoised_Cxy.shape)
```

## Testing

To run the comprehensive tests, including a training simulation:

```bash
# Ensure environment is active
conda activate ccc_env
python -m unittest tests/test_model.py
```

## Requirements

The codebase relies on the following core libraries:

*   `tensorflow>=2.10.0`
*   `keras`

Install them via:
```bash
pip install -r requirements.txt
```

## Citation
[Insert Citation Here]
