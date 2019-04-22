import tensorflow as tf
import os

##########################################
# -> Custom Dataset for simplicity of code
##########################################

class Dataset():
    def __init__(self):
        self.batch_size = 0

        # 1. check and download datset if needed
        self.dataset_dir = self.download_dataset()
        self.dir_list, self.label_list = self.get_list()

    def get(self, batch_size):
        self.batch_size = batch_size
        ds = tf.data.Dataset.from_tensor_slices((self.dir_list, self.label_list))

        ds = ds.map(self.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.shuffle(buffer_size=self.size())
        ds = ds.repeat()
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)  # Let's dataset fetch file while training
        ds = iter(ds)

        return ds

    def size(self):
        return len(self.dir_list)

    def preprocess_img(self, image):  # for preprocessing
        image = tf.image.decode_jpeg(image, channels=3)  # decode jpeg file
        image = tf.image.resize(image, [192, 192])  # resize to 192
        image /= 255.0  # normalize to [0,1] range

        return image

    def load_image(self, path, label):
        label = label

        image = tf.io.read_file(path)  # read from path
        image = self.preprocess_img(image)  # pre-process

        return label, image

    def download_dataset(self):
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

        return dataset_dir

    def get_list(self):
        classes = [name for name in os.listdir(self.dataset_dir) if os.path.isdir(os.path.join(self.dataset_dir, name))]

        # 2) make list of directories
        dir_list = []
        label_list = []
        self.class_id = {}

        for idx, class_name in enumerate(classes):
            self.class_id[class_name] = idx

        for idx, class_name in enumerate(classes):
            for file in os.listdir(os.path.join(self.dataset_dir, class_name)):
                temp_dir = os.path.join(self.dataset_dir, class_name, file)
                dir_list.append(temp_dir)

                temp_label = idx
                label_list.append(temp_label)

        return dir_list, label_list

if __name__ == '__main__':
    dataset = Dataset()
    ds = dataset.get(batch_size=32)

