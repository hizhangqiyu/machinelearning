# MIT License
#
# Copyright (c) 2016 AppleFairy
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import tensorflow as tf
import pandas as panda
import numpy as numpy
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

LABLE_KIND_NUM = 10
BATCH = 100			# N 	batch size.
CHANNEL = 1			# C 	channel number.
IMAGE_HEIGHT = 28	# H
IMAGE_WIDTH = 28	# W

KERNEL_HEIGHT = 5
KERNEL_WIDTH = 5

STEP_NUM = 20000
VALID_DATA_SIZE = 10000
LR = 0.001			# learnning rate.

def generate_data():
	data = pd.read_csv('train.csv')
	label = np.array(data.pop('label'))
	label = LabelEncoder().fit_transform(label)[:, None]
	label = OneHotEncoder().fit_transform(label).todense()

	data = StandardScaler().fit_transform(np.float32(data.values))
	data = data.reshape(-1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL)
	train_data, valid_data = data[:-VALID_DATA_SIZE], data[-VALID_DATA_SIZE:]
	train_label, valid_label = label[:-VALID_DATA_SIZE], label[-VALID_DATA_SIZE:]

	test = pd.read_csv('test.csv')
	test_data = StandardScaler().fit_transform(np.float32(test.values))
	test_data = test_data.reshape(-1, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL)

	return train_data, train_label, valid_data, valid_label, test_data

def weight_variable(shape):
  	initial = tf.truncated_normal(shape, stddev=0.1)
  	return tf.Variable(initial)

def bias_variable(shape):
  	initial = tf.constant(0.1, shape=shape)
  	return tf.Variable(initial)

def conv2d(x, W):
	# data_format: Defaults to NHWC, cudnn defaults tto NCHW.
	# use_cudnn_on_gpu: Defaults to true.
	# strides: 1-D of length 4. The stride of sliding window for each dimension of input.
  	return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	# value: 4-D tensor.
	# ksize: A list of ints that has length >= 4. The size of the window for each dimension of the input tensor.
	# strides: A list of ints that has length >= 4. The stride of the sliding window for each dimension of the input tensor.
  	return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def network():
	x_image = tf.reshape(x, [-1,28,28,1])

	# converlution layer 1.
	# kernel 5x5, 1 input channel, 32 output channel. 
	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)		# 28x28 => 14x14

	# converlution layer 2.
	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)		# 14x14 => 7x7

	# droup out/densely connected layer.
	W_fc1 = weight_variable([7 * 7 * 64, 1024])
	b_fc1 = bias_variable([1024])
	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	# drop out.
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	# fully connected layer.
	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])
	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	return y_conv


def main():
	train_data, train_label, valid_data, valid_label, test_data = generate_data()

	tf_data = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL])
	tf_label = tf.placeholder(tf.float32, shape=[None, LABLE_KIND_NUM])

	# Prediction:
	tf_pred = tf.nn.softmax(network(tf_data))
	tf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network(tf_data),
    	                                                             labels=tf_label))
	tf_acc = 100*tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(tf_pred, 1), tf.argmax(tf_label, 1))))

	#tf_opt = tf.train.GradientDescentOptimizer(LR)
	#tf_opt = tf.train.AdamOptimizer(LR)
	tf_opt = tf.train.RMSPropOptimizer(LR)
	tf_step = tf_opt.minimize(tf_loss)

	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())

	ss = ShuffleSplit(n_splits=STEP_NUM, train_size=BATCH)
	# ss.get_n_splits(train_data, train_label)
	history = [(0, np.nan, 10)]
	for step, (idx, _) in enumerate(ss.split(train_data,train_label), start=1):
    	fd = {tf_data:train_data[idx], tf_labels:train_labels[idx]}
    	session.run(tf_step, feed_dict=fd)
    	if step%500 == 0:
        	fd = {tf_data:valid_data, tf_label:valid_label}
        	valid_loss, valid_accuracy = session.run([tf_loss, tf_acc], feed_dict=fd)
        	history.append((step, valid_loss, valid_accuracy))
        	print('Step %i \t Valid. Acc. = %f'%(step, valid_accuracy))

	test_pred = session.run(tf_pred, feed_dict={tf_data:test_data})
	test_labels = np.argmax(test_pred, axis=1)

	submission = pd.DataFrame(data={'ImageId':(np.arange(test_labels.shape[0])+1), 'Label':test_labels})
	submission.to_csv('submission.csv', index=False)
	submission.tail()



