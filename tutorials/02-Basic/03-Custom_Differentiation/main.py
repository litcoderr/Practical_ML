import tensorflow as tf

# ================================================================== #
#                         Table of Contents                          #
# ================================================================== #

# 1. Usage of Gradient Tape
# 2. Control Flow of Gradient Tape

# =================================================================== #

if __name__ == '__main__':
    # ================================= #
    # 1. Use of Gradient Tape           #
    # ================================= #

    # 1) If Using Variable which is tracked by default

    x = tf.Variable(1.0)

    with tf.GradientTape() as t:
        with tf.GradientTape() as t2:
            y = x * x * x
        dy_dx = t2.gradient(y, x)
    d2y_dx2 = t.gradient(dy_dx, x)

    print(dy_dx)
    print(d2y_dx2)

    # 2) If using random tensor
    x = tf.constant([3.0])
    with tf.GradientTape(persistent=True) as t:  # Use persistent=True when needing more than one gradient
        t.watch(x)  # You need to watch that tensor
        y = x * x
        z = y * y

    dz_dx = t.gradient(z, x)
    dy_dx = t.gradient(y, x)
    del t  # Make sure to drop the reference to the tape

    print(dz_dx)
    print(dy_dx)

    # ========================================= #
    # 2. Control Flow of Gradient Tape          #
    # ========================================= #

    # 1) Define Graph
    def my_small_computaion(x, y):
        output = 1.0
        for _ in range(y):
            output = tf.multiply(output, x)

        return output

    # 2) Define Grad Computation Function
    def grad(x, y):
        with tf.GradientTape() as t:
            t.watch(x)
            output = my_small_computaion(x, y)
        gradient = t.gradient(output, x)

        return gradient

    # 3) Compute Gradient of x !!
    input = tf.constant([3.0])
    g = grad(input, 6)
    print("How Control Flow works, grad : {}".format(g))
