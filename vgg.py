# Adapted from https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/models/image/cifar10/cifar10.py

from datetime import datetime
import math
import time
import numpy as np
import dataset
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
        kernel = tf.Variable(tf.truncated_normal([n_in, n_out], dtype=tf.float32, stddev=1e-1),  name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=tf.float32), name='biases')
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
    onehot_labels = tf.sparse_to_dense(concated, tf.pack([batch_size, 10]), 1.0, 0.0)
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
    affn2 = affine_op(affn1,  name="affn2", n_in=4096, n_out=10)
    affn3 = affine_op(affn2,  name="affn3", n_in=10, n_out=10)
    return affn3
    # return affn1


def train(lr=0.001, max_step=1000):
    """
    train model

    :param lr:  This is the learning rate
    """
    with tf.Graph().as_default():
        # Generate some dummy images.
        # image_size = 224
        # images = tf.Variable(tf.random_normal([batch_size, image_size, image_size, 3],
        #                                       dtype=tf.float32,
        #                                       stddev=1))
        #
        # labels = tf.Variable(tf.ones([batch_size], dtype=tf.int32))

        in_images = tf.placeholder("float", [batch_size, 32, 32, 3])
        images = tf.image.resize_images(in_images, 224, 224)
        labels = tf.placeholder("int32", [batch_size])

        # Build a Graph that computes the logits predictions from the
        # inference model.
        last_layer = inference(images)



        # Add a simple objective so we can calculate the backward pass.
        objective = loss(last_layer, labels)
        optimizer = tf.train.AdagradOptimizer(lr)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_step = optimizer.minimize(objective)

        # grab variables we want to log
        tf.scalar_summary("loss", objective)

        summaries = tf.merge_all_summaries()

        # Build an initialization operation.
        initializer = tf.initialize_all_variables()

        # Start running operations on the Graph.
        with tf.Session() as sess:
            sess.run(initializer)
            writer = tf.train.SummaryWriter("train_logs", graph_def=sess.graph_def)
            for i in range(max_step):
                trn, tst = dataset.get_cifar10(batch_size)
                for batch in trn:
                    X = np.vstack(batch[0]).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                    Y = np.array(batch[1])
                    result = sess.run(
                        [train_step, summaries, objective],
                        feed_dict = {
                            in_images: X,
                            labels: Y
                        }
                    )
                    writer.add_summary(result[1], i)
                    print i, result[2]
                    if result[2] is np.NaN:
                        return





if __name__ == '__main__':
    train()
