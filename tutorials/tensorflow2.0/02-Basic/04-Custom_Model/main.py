import tensorflow as tf

# ================================================================== #
#                         Table of Contents                          #
# ================================================================== #

# 1. Non-Custom Layers
# 2. Custom Layers
# 3. Custom Model

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

    def get_config(self):
        config = super(CustomLayer, self).get_config()
        config.update({'units': self.n_output})
        return config


custom_layer_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    CustomLayer(10),
    tf.keras.layers.ReLU()
])

# 3. Custom Model
class CustomModel(tf.keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.flatten_layer = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.custom_layer = CustomLayer(10)
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs):
        output = self.flatten_layer(inputs)
        output = self.custom_layer(output)
        output = self.relu(output)
        return output


if __name__ == '__main__':
    input = tf.Variable(tf.ones((1, 28, 28)))

    # 1. Running Non-Custom Layers
    output = non_custom_model(input)
    print("non_custom_model_result: {}".format(output.shape))

    # 2. Running Custom Layers
    output = custom_layer_model(input)
    print("output of custom layer model : {}".format(output.shape))
    print("config : {}".format(custom_layer_model.get_config()))

    # 3. Running Custom Model
    model = CustomModel()
    output = model(input)
    print("output of custom model : {}".format(output.shape))

    print(model.summary())
