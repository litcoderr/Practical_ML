import tensorflow as tf
import numpy as np

# ================================================================== #
#                         Table of Contents                          #
# ================================================================== #

# 1. Basic Tensor Operations (with Eager Execution)
# 2. Numpy Compatibility
# 3. Device Management

# =================================================================== #

if __name__ == '__main__':

    # ================================= #
    # 1. Basic Tensor Operations        #
    # ================================= #

    # 1-1) Generate Custom Tensor
    a = tf.Variable([1.0, 4.0, 3.5])
    b = tf.Variable([3.0, 2.4, 1.5])

    # 1-2) Generate Random Tensor
    # generated random tensor shape of (5,2) ranging from 0 to 10
    c = tf.Variable(tf.random.uniform([5, 2], 0, 10, dtype=tf.int32, seed=0))

    # 1-3) Addition
    print("1. a+b = {}".format(a + b))

    # 1-4) Subtraction
    print("2. a-b = {}".format(a - b))

    # 1-5) Element-wise Multiplication
    print("3. a*b = {}".format(a * b))

    # 1-6) Element-wise Division
    print("4. a/b = {}".format(a / b))

    # 1-7) Exponential Operations
    print("5. a**2 = {}".format(a ** 2))
    print("--------------------------------------")
    # ================================= #
    # 2. Numpy Compatibility            #
    # ================================= #

    array = np.ones([3, 3])
    tensor = tf.multiply(array, 21)  # tensor operation with numpy array
    print("Numpy arrays are automatically converted\n{}".format(tensor))

    np_op = np.add(tensor,10)
    print("Numpy Operations with Tensors are possible\n{}".format(np_op))

    np_converted = tensor.numpy()
    print("Tensors can be easily converted to numpy\n{}".format(np_converted))
    print("--------------------------------------")
    # ================================= #
    # 3. Device Management              #
    # ================================= #

    # 3-1) Check if GPU is available
    print("Is GPU Available? : {}".format(tf.test.is_gpu_available()))

    # 3-2) Check where tensor is located
    x = tf.random.uniform([3, 3])
    print("where is x located? : {}".format(x.device))

    # 3-3) Explicit placement to certain device
    def operation(x):
        for _ in range(10):
            x = x*x
        return x

    # Running on CPU
    with tf.device("CPU:0"):
        x = tf.random.uniform([3, 3], 0, 1)
        x = operation(x)
        print("Ran on cpu : {}".format(x.device))
        print("result : {}".format(x))

    # Running on GPU if available
    if tf.test.is_gpu_available():
        with tf.device("GPU:0"):
            x = tf.random.uniform([3, 3], 0, 10)
            x = operation(x)
            print("Ran on gpu : {}".format(x.device))
            print("result : {}".format(x))
