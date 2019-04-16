import tensorflow as tf

# ================================================================== #
#                         Table of Contents                          #
# ================================================================== #

# 1. Non-Custom Layers
# 2. Custom Layers

# =================================================================== #

# 1. Non-Custom Layers
non_custom_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])


# 2. Custom Layers
class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, num_output):
        super(CustomLayer, self).__init__()
        self.n_output = num_output

    def build(self, input_shape):
        self.kernel = self.add_variable("kernel",
                                        shape=[int(input_shape[-1]), self.n_output])

    def call(self, input):
        return tf.matmul(input, self.kernel)


custom_layer_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    CustomLayer(10),
    tf.keras.layers.ReLU()
])

# 3. Custom Model


if __name__ == '__main__':
    input = tf.Variable(tf.ones((1, 28, 28)))

    # 1. Running Non-Custom Layers
    output = non_custom_model(input)
    print("non_custom_model_result: {}".format(output.shape))

    # 2. Running Custom Layers
    output = custom_layer_model(input)
    print("output of custom layer model : {}".format(output.shape))
