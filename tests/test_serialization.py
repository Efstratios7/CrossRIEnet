import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf

from crossrie import CrossRIELayer


def test_crossrie_model_keras_roundtrip_preserves_outputs(tmp_path):
    tf.keras.utils.set_random_seed(11)

    input_cxx = tf.keras.Input(shape=(None, None), name="Cxx")
    input_cyy = tf.keras.Input(shape=(None, None), name="Cyy")
    input_cxy = tf.keras.Input(shape=(None, None), name="Cxy")
    input_t = tf.keras.Input(shape=(), name="T_samples")
    output = CrossRIELayer(
        encoding_units=[3],
        lstm_units=[3],
        final_hidden_layer_sizes=[2],
        outputs=["Cxy", "Sxy"],
    )([input_cxx, input_cyy, input_cxy, input_t])
    model = tf.keras.Model(
        inputs=[input_cxx, input_cyy, input_cxy, input_t],
        outputs=output,
    )

    inputs = [
        tf.eye(2, batch_shape=[1], dtype=tf.float32),
        tf.eye(3, batch_shape=[1], dtype=tf.float32),
        tf.random.normal((1, 2, 3), dtype=tf.float32),
        tf.constant([10.0], dtype=tf.float32),
    ]
    before = model(inputs, training=False)

    path = tmp_path / "crossrie_roundtrip.keras"
    model.save(path)
    loaded = tf.keras.models.load_model(path)
    after = loaded(inputs, training=False)

    assert set(after.keys()) == {"Cxy", "Sxy"}
    assert after["Cxy"].shape == before["Cxy"].shape
    assert after["Sxy"].shape == before["Sxy"].shape
    assert np.all(np.isfinite(after["Cxy"].numpy()))
    assert np.all(np.isfinite(after["Sxy"].numpy()))
    np.testing.assert_allclose(after["Cxy"].numpy(), before["Cxy"].numpy(), atol=0)
    np.testing.assert_allclose(after["Sxy"].numpy(), before["Sxy"].numpy(), atol=0)
