# Adapted from https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/models/image/cifar10/cifar10.py

from datetime import datetime
import math
import time

import tensorflow.python.platform
import tensorflow as tf


batch_size = 2

def conv_op(input_op, name, kw, kh, n_in, n_out, dw, dh):
    with tf.name_scope(name) as scope:
        kernel_init_val = tf.truncated_normal([kh, kw, n_in, n_out], dtype=tf.float32, stddev=1e-1)
        kernel = tf.Variable(kernel_init_val, trainable=True, name='w')
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        activation = tf.nn.relu(z, name=scope)
        return activation

def affine_op(input_op, name, n_in, n_out):
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal([n_in, n_out], dtype=tf.float32, stddev=1e-1),  trainable=True, name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=tf.float32), trainable=True, name='biases')
        affine1 = tf.nn.relu_layer(input_op, kernel, biases, name=name)
        return affine1

def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='VALID',
                          name=name)

def loss(logits, labels):
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat(1, [indices, labels])
    onehot_labels = tf.sparse_to_dense(concated, tf.pack([batch_size, 1000]), 1.0, 0.0)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, onehot_labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss


def inference(input_op):
    conv1 = conv_op(input_op, name="conv1", kh=3, kw=3, n_in=3, n_out=64, dh=1, dw=1)
    pool1 = mpool_op(conv1,   name="pool1", kh=2, kw=2, dw=2, dh=2)
    conv2 = conv_op(pool1,    name="conv2", kh=3, kw=3, n_in=64, n_out=128, dh=1, dw=1)
    pool2 = mpool_op(conv2,   name="pool2", kh=2, kw=2, dh=2, dw=2)
    conv3 = conv_op(pool2,    name="conv3", kh=3, kw=3, n_in=128, n_out=512, dh=1, dw=1)
    conv4 = conv_op(conv3,    name="conv4", kh=3, kw=3, n_in=512, n_out=512, dh=1, dw=1)
    pool4 = mpool_op(conv4,   name="pool4", kh=2, kw=2, dh=2, dw=2)
    conv5 = conv_op(pool4,    name="conv5", kh=3, kw=3, n_in=512, n_out=512, dh=1, dw=1)
    conv6 = conv_op(conv5,    name="conv6", kh=3, kw=3, n_in=512, n_out=512, dh=1, dw=1)
    pool6 = mpool_op(conv6,   name="pool6", kh=2, kw=2, dh=2, dw=2)
    conv7 = conv_op(pool6,    name="conv7", kh=3, kw=3, n_in=512, n_out=512, dh=1, dw=1)
    conv8 = conv_op(conv7,    name="conv8", kh=3, kw=3, n_in=512, n_out=512, dh=1, dw=1)
    pool8 = mpool_op(conv8,   name="pool8", kh=2, kw=2, dw=2, dh=2)
    resh1 = tf.reshape(pool8, [-1,512*7*7], name="resh1")
    affn1 = affine_op(resh1,  name="affn1", n_in=512*7*7, n_out=4096)
    affn2 = affine_op(affn1,  name="affn2", n_in=4096, n_out=4096)
    affn3 = affine_op(affn2,  name="affn3", n_in=4096, n_out=1000)
    return affn3

def predict():
    pass

def train(lr=0.01):
    """
    train model

    :param lr:  This is the learning rate
    """
    with tf.Graph().as_default():
        # Generate some dummy images.
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size,
                                               image_size,
                                               image_size, 3],
                                              dtype=tf.float32,
                                              stddev=1e-1))

        labels = tf.Variable(tf.ones([batch_size], dtype=tf.int32))

        # Build a Graph that computes the logits predictions from the
        # inference model.
        last_layer = inference(images)


        # Build an initialization operation.
        initializer = tf.initialize_all_variables()

        # Start running operations on the Graph.
        sess = tf.Session()
        sess.run(initializer)

        # Add a simple objective so we can calculate the backward pass.
        objective = loss(last_layer, labels)
        train_step = tf.train.GradientDescentOptimizer(lr).minimize(objective)
        for i in range(1000):
            print i
            sess.run(train_step)
        sess.close()




if __name__ == '__main__':
    train()
