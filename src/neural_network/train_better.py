import tensorflow as tf
from dataset_api_wrapper.dataset_helper import images_from_list_files
from augmentation.augmentations import image_net_normalization
from PIL import Image

batch_size = 4


def small_net(inputs):
    net = tf.add(inputs, tf.constant(25, dtype=tf.float32), name='addition')

    return net


def load_dataset():
    list_files = ['D:\Data\python\PrakikumML2017\src\datasets\isic_image_data.dataset']  # ,
    # '/lhome/joeinig/datasets/cspub_val_gtFine.dataset']

    dataset = images_from_list_files(list_files, num_threads=None, output_buffer_size=1,
                                     augmentations=[image_net_normalization])
    dataset = dataset.batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()


def main():
    total_iterations = 1000

    inputs = load_dataset()
    with tf.name_scope("input"):
        x = tf.reshape(tf.to_float(inputs[0]), [-1, 299, 299, 3], name="input_reshape")
        # y = tf.reshape(inputs[1], [-1, 2])

    net = small_net(x)

    print('setup graph')
    with tf.Session() as sess:
        print('init sess')
        # initialize variables BEFORE loading existing weights
        sess.run(tf.global_variables_initializer())
        print('init vars')
        for i in range(total_iterations):
            print('iteration: ' + str(i))
            # result = sess.run(x)
            result = sess.run(inputs)
            Image.fromarray(result).show()
            input()


if __name__ == "__main__":
    main()
