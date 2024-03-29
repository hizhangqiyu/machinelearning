import tensorflow as tf
import os

IMAGE_SIZE = 24

NUM_CLASSES = 10
NUM_EXAMPLES_PRE_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PRE_EPOCH_FOR_EVAL = 10000

def read_cifar10(filename_queue):
    class CIFAR10Record():
        pass
    result = CIFAR10Record()

    '''
        cifar-10 image data format:
        1. each 1+(32x32x3) bytes is one image.
        2. the first 1 byte is label of this image(10 classes in all).
        3. the next 1024 bytes are red channel values, the next 1024 bytes
           are green channel values, the last 1024 bytes are blue channel
           values and then the later data is the next image.
    '''
    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    record_bytes = label_bytes + image_bytes

    record_bytes = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)    

    record_bytes = tf.decode_raw(value, tf.uint8)

    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], 
        [label_bytes + image_bytes]), [result.depth, result.height, resutl.width])

    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result

def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    tf.summary.image('images', images)

    return images, tf.reshape(label_batch, [batch_size])

def distorted_inputs(data_dir, batch_size):
     # cifar-10 has 5 data files.
    filenames = [op.path.join(data_dir, 'data_batch_%d.bin' % i for i in range(1,6))]

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file:' + f)

    filename_queue = tf.train.string_input_producer(filenames)

    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    # ?????
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

    float_image = tf.image.per_image_standardization(distorted_image)

    float_image.set_shape([height, width, 3])
    read_imput.label.set_shape([1])

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PRE_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
      
    print ('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

    return _generate_image_and_label_batch(floa_image,
        read_input.label, min_queue_examples, batch_size, shuffle=True)

def inputs(eval_data, data_dir, batch_size):
    if not eval_data:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i for i in range(1, 6))]
        num_examples_pre_epoch = NUM_EXAMPLES_PRE_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]
        num_examples_pre_epoch = NUM_EXAMPLES_PRE_EPOCH_FOR_EVAL

    for f in filemames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    filename_queue = tf.train.string_input_producer(filenames)

    read_input = read_cifar10(filename_queue)
    reshape_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    resized_image = tf.image.resize_image_with_crop_or_pad(reshape_image, height, width)

    float_image = tf.image.per_image_standardization(resized_image)

    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_pre_epoch * min_fraction_of_examples_in_queue)

    return _generate_image_and_label_batch(float_image, read_input.label,
        min_queue_examples, batch_size, shuffle=False)