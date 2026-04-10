import tensorflow as tf

class DummyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(5, use_bias=False)
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None], dtype=tf.float32)])
    def call_with_grad(self, x):
        with tf.GradientTape() as tape:
            y = x * self.dense.kernel[0, 0]
            # fake svd
            W1 = tf.ones_like(y)
            V2, _ = tf.linalg.qr(W1, full_matrices=False)
            loss = tf.reduce_sum(V2)
        grads = tape.gradient(loss, self.trainable_variables)
        return grads
        
m = DummyModel()
m.dense.build((None, 5))
m.call_with_grad(tf.random.normal((2,4,6)))
print("Done")
