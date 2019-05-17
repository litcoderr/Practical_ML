import tensorflow as tf
import tensorflow.keras.layers as l
import data

batch_size = 16

# 1. Make Model
class MyModel(tf.keras.Model):  # Some Fully connected Neural Network
    def __init__(self):
        super(MyModel, self).__init__()
        self.flatten = l.Flatten()
        self.linear1 = l.Dense(100, activation="tanh")
        self.linear2 = l.Dense(10, activation="tanh")
        self.linear3 = l.Dense(5)

    def call(self, inputs):
        output = self.flatten(inputs)
        output = self.linear1(output)
        output = self.linear2(output)
        output = self.linear3(output)
        return output

# 2. Define Loss function
def loss_func(predicted, ground_truth):
    return tf.nn.softmax_cross_entropy_with_logits(predicted, ground_truth)

# 3. Get Training Data
# Custom Dataset Class --> data.py
dataset = data.Dataset()  # Make dataset object
ds = dataset.get(batch_size=batch_size)  # get dataset interator object with 32 batch size

# 4. Define Train Function
def train(model, x, y, n_classes):
    with tf.GradientTape() as t:
        y_ = model(x)
        answer = tf.argmax(y_, axis=1)
        print("{}\n{}".format(y, answer))
        loss = loss_func(y_, tf.one_hot(y, n_classes))

        return loss, t.gradient(loss, model.trainable_variables)


if __name__ == '__main__':
    is_Train = True

    n_epochs = 10
    lr = 0.01

    size = dataset.size()
    n_classes = len(dataset.class_id)

    model = MyModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    train_loss_results = []

    if is_Train:
        # 5. Start Training Sequence
        for epoch in range(n_epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()
            for iteration in range(int(size/batch_size)):
                label, image = next(ds)
                loss, gradient = train(model, image, label, n_classes)
                epoch_loss_avg(loss)
                print("epoch: {} [{}/{}] loss: {}".format(epoch+1, iteration, size/batch_size, epoch_loss_avg.result()))
                optimizer.apply_gradients(zip(gradient, model.trainable_variables))

            train_loss_results.append(epoch_loss_avg.result())

    else:
        model.build((192, 192))
        print(model.summary())
