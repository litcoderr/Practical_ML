import tensorflow as tf
import os

import matplotlib.pyplot as plt

# ================================================================== #
#                         Table of Contents                          #
# ================================================================== #

# 1. Prepare Files to Read
# 2. Generate list of directories to read from
# 3. Make loading function
# 4. Make tf.data.Dataset
# 5. pipe data to model
# 6. Tips and Tricks (Batches, Shuffles, Repeats etc)

# =================================================================== #


def preprocess_img(image):  # for preprocessing
    image = tf.image.decode_jpeg(image, channels=3)  # decode jpeg file
    image = tf.image.resize(image, [192, 192])  # resize to 192
    image /= 255.0  # normalize to [0,1] range

    return image


def load_image(path):
    label = path[0]
    image = tf.io.read_file(path[1])  # read from path
    image = preprocess_img(image)  # pre-process

    return label, image


if __name__ == '__main__':
    # ================================= #
    # 1. Prepare Files to Read          #
    # ================================= #

    # 0) Prepare Directory to save images
    cur_dir = os.path.dirname(os.path.realpath(__file__))  # current directory of file
    root = os.path.join(cur_dir, "dataset")  # root directory of dataset
    if not os.path.isdir(root):  # if not exist --> make one
        os.mkdir(root)

    # 1) Download Flower Images!!
    # --> It is a big file. Have a cup of coffee and relax for a few minutes
    dataset_dir = os.path.join(root, "datasets", "flower_photos")
    if not os.path.isdir(dataset_dir):
        dataset_dir = tf.keras.utils.get_file(
            origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
            fname='flower_photos', untar=True, cache_dir=root)

    print("Directory to Flower Images: {}".format(dataset_dir))

    # ============================================= #
    # 2. Generate list of directories to read from  #
    # ============================================= #

    # 1) get classes
    classes = [name for name in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, name))]

    # 2) make list of directories
    dir_list = []

    for class_name in classes:
        for file in os.listdir(os.path.join(dataset_dir, class_name)):
            temp = [class_name, os.path.join(dataset_dir, class_name, file)]
            dir_list.append(temp)

    print("Classes : {}".format(classes))
    print("File List : {}".format(dir_list))

    # ============================================= #
    # 3. Make loading function                      #
    # ============================================= #

    #                                                         /\
    # Loading function/ Pre-process function is defined above ||
    #                                                         ||

    # ============================================= #
    # 4. Make tf.data.Dataset                       #
    # ============================================= #

    ds = tf.data.Dataset.from_tensor_slices(dir_list) # Make Dataset from tensor slices
    image_ds = ds.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # ============================================= #
    # 5. Pipe data to model                         #
    # ============================================= #

    for idx, (label, image) in enumerate(image_ds.take(10)):
        print("[{}] {}".format(idx + 1, label))
        plt.imshow(image)
        plt.show()

    # ============================================= #
    # 6. Tips and Tricks                            #
    # ============================================= #

    BATCH_SIZE = 32  # Batch size
    N = len(dir_list)  # number of images

    image_ds = image_ds.shuffle(buffer_size=N)
    image_ds = image_ds.repeat()
    image_ds = image_ds.batch(BATCH_SIZE)
    image_ds = image_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  # Let's dataset fetch file while training
    image_ds = iter(image_ds)

    for idx in range(10):
        label, image = next(image_ds)
        temp_label = label[0]  # take first item in batch
        temp_image = image[0]  # take first item in batch

        print("[{}] {}".format(idx+1, temp_label))
        plt.imshow(temp_image)
        plt.show()


