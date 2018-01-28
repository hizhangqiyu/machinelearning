import tensorflow as tf
import gzip
import os
import tempfile
import time
import sys
import argparse
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import mnist

FLAGS = None

def placeholder_inputs(batch_size):
    images_palceholder = tf.placeholder(tf.float32, shape=[batch_size, mnist.IMAGE_PIXELS])
    labels_placeholder = tf.placeholder(tf.int32, shape=[batch_size])
    return images_palceholder, labels_placeholder

def fill_feed_dict(data_set, images_pl, labels_pl):
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size, FLAGS.fake_data)
    feed_dict = {images_pl: images_feed, labels_pl: labels_feed}
    return feed_dict

def do_eval(sess, eval_correct, images_palceholder, labels_placeholder, data_set):
    true_count = 0
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size

    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set, images_palceholder, labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)

    precision = float(true_count) / num_examples
    print('Num examples: %d Num correct: %d Precision: %0.04f' % (num_examples, true_count, precision))

def run_trainning():
    data_sets = read_data_sets('../../../dataset/mnist', FLAGS.fake_data)

    with tf.Graph().as_default():
        images_palceholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)
        logits = mnist.inference(images_palceholder, FLAGS.hidden1, FLAGS.hidden2)
        loss = mnist.loss(logits, labels_placeholder)
        train_op = mnist.training(loss, FLAGS.learning_rate)
        eval_correct = mnist.evaluation(logits, labels_placeholder)
        
        summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.Session()
        summary_writer = tf.summary.FileWriter('../../../dataset/tensorboard', sess.graph)

        sess.run(init)

        for step in range(FLAGS.max_steps):
            start_time = time.time()

            feed_dict = fill_feed_dict(data_sets.train, images_palceholder, labels_placeholder)
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            duration = time.time() - start_time

            if step % 100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))

                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_file = '../../../dataset/tensorboard/model.ckpt'
                saver.save(sess, checkpoint_file, global_step=step)
                
                print('Training Data Eval:')
                do_eval(sess, eval_correct, images_palceholder, labels_placeholder, data_sets.train)
                
                print('Validation Data Eval:')
                do_eval(sess, eval_correct, images_palceholder, labels_placeholder, data_sets.validation)
                
                print('Test Data Eval:')
                do_eval(sess, eval_correct, images_palceholder, labels_placeholder, data_sets.test)

def main(_):
    run_trainning()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--max_steps', type=int, default=200, help='Number of steps to run trainer.')
    parser.add_argument('--hidden1', type=int, default=128, help='Number of units in hidden layer 1.')
    parser.add_argument('--hidden2', type=int, default=32, help='Number of units in hidden layer 2.')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--input_data_dir', type=str, default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
        'tensorflow/mnist/input_data'), help='Directory to put the input data.')
    parser.add_argument('--log_dir', type=str, default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
        'tensorflow/mnist/logs'), help='Directory to put the log data.')
    parser.add_argument('--fake_data', default=False, help='If true, uses fake data for unit testing.', action='store_true')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)