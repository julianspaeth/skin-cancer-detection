import tensorflow as tf
import tensorflow.contrib.data as tfdata


def image_data(dataset, augmentations=None,
               num_threads=None, output_buffer_size=None,
               shuffle_buffer_size=None):
    """
        Args:
            dataset: The data set that contains the list of paths.
            augmentations: A list of functions that take a structure of images and label information and return a
                structure of images and label information.
            num_threads: The number of threads to use for loading and augmenting the images. `None` ==  single
            output_buffer_size: The size of the output buffer that is used for mapping the image
                decoder and augmentations over the dataset.
            shuffle_buffer_size: The size of the shuffle buffer that is used to shuffle the data
                files
        Returns:
            Data set of tuples where each component is an image from the corresponding file.
    """
    if augmentations is None:
        augmentations = []

    def _read_image(img):
        file_string = tf.read_file(img)
        return tf.image.decode_image(file_string)

    def _read_json(img):
        file_string = tf.read_file(img)

        # TOdo read json in tensorflow correctly
        return



    def _read_images(*args):
        images = tuple([_read_image( args[0]), _read_json(args[1]) ])
        for a in augmentations:
            images = a(images)
        return images

    if shuffle_buffer_size is not None:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.repeat(-1)  # infinite repeat

    return dataset.map(_read_images,
                       num_threads=num_threads,
                       output_buffer_size=output_buffer_size)


def map(f, data):
    """
        Performs a map over lists of python and applies f on each element
    """
    if isinstance(data, list):
        return [map(f, x) for x in data]
    else:
        return f(data), f(data)


def images_from_list_files(file_paths, **kwargs):
    """
        Takes a structure of paths to files that contain lists of images and loads these.
    """
    datasets = map(tfdata.TextLineDataset, file_paths)
    if isinstance(datasets, list):
        datasets = tuple(datasets)
    dataset = tfdata.Dataset.zip(datasets)
    print(dataset.output_shapes)
    return image_data(dataset, **kwargs)
