from datetime import datetime
import math
import time
import numpy as np
import dataset
import tensorflow.python.platform
import tensorflow as tf


def conv_op(input_op, name, kw, kh, n_out, dw, dh):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w",
                                 shape=[kh, kw, n_in, n_out],
                                 dtype=tf.float32, 
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        activation = tf.nn.relu(z, name=scope)
        return activation

def fc_op(input_op, name, n_out):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+"w",
                                 shape=[n_in, n_out],
                                 dtype=tf.float32, 
                                 initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        return activation

def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='SAME',
                          name=name)

def loss_op(logits, labels, batch_size):
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat(1, [indices, labels])
    onehot_labels = tf.sparse_to_dense(concated, tf.pack([batch_size, 10]), 1.0, 0.0)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, onehot_labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss



def evaluate_op(predictions, labels):
    """Evaluate the quality of the predictions at predicting the label.

    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size], with values in the range [0, NUM_CLASSES).
    Returns:
        A scalar int32 tensor with the number of examples (out of batch_size)
        that were predicted correctly.

    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label's is was in the top k (here k=1)
    # of all logits for that example.
    correct = tf.nn.in_top_k(predictions, labels, 1)

    # Return the number of true entries.
    total_correct = tf.reduce_sum(tf.cast(correct, tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy, total_correct

def inference_op(input_op, training=False):

    dropout_keep_prob = 0.4 if training else 1.0
    # assume input_op shape is 224x224x3

    # block 1 -- outputs 112x112x64
    conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1)
    conv1_2 = conv_op(conv1_1,  name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1)
    pool1 = mpool_op(conv1_2,   name="pool1",   kh=2, kw=2, dw=2, dh=2)

    # block 2 -- outputs 56x56x128
    conv2_1 = conv_op(pool1,    name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1)
    conv2_2 = conv_op(conv2_1,  name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1)
    pool2 = mpool_op(conv2_2,   name="pool2",   kh=2, kw=2, dh=2, dw=2)

    # # block 3 -- outputs 28x28x256
    conv3_1 = conv_op(pool2,    name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1)
    conv3_2 = conv_op(conv3_1,  name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1)
    pool3 = mpool_op(conv3_2,   name="pool3",   kh=2, kw=2, dh=2, dw=2)

    # block 4 -- outputs 14x14x512
    conv4_1 = conv_op(pool3,    name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1)
    conv4_2 = conv_op(conv4_1,  name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1)
    conv4_3 = conv_op(conv4_2,  name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1)
    pool4 = mpool_op(conv4_3,   name="pool4",   kh=2, kw=2, dh=2, dw=2)

    # block 5 -- outputs 7x7x512
    conv5_1 = conv_op(pool4,    name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1)
    conv5_2 = conv_op(conv5_1,  name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1)
    conv5_3 = conv_op(conv5_2,  name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1)
    pool5 = mpool_op(conv5_3,   name="pool5",   kh=2, kw=2, dw=2, dh=2)

    # flatten
    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1")

    # fully connected
    fc6 = fc_op(resh1, name="fc6", n_out=1000)
    fc6_drop = tf.nn.dropout(fc6, dropout_keep_prob, name="fc6_drop")

    fc7 = fc_op(fc6_drop, name="fc7", n_out=50)
    fc7_drop = tf.nn.dropout(fc7, dropout_keep_prob, name="fc7_drop")

    fc8 = fc_op(fc7_drop, name="fc8", n_out=10)
    softmax = tf.nn.softmax(fc8)
    predictions = tf.argmax(softmax, 1)
    return predictions, softmax, fc8

def inference_cifar10_vgg(input_op, training=False):

    dropout_keep_prob = 1.0 if training else 1.0
    # assume input_op shape is 224x224x3

    # block 1 -- outputs 112x112x64
    conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=16, dh=1, dw=1)
    conv1_2 = conv_op(conv1_1,  name="conv1_2", kh=3, kw=3, n_out=16, dh=1, dw=1)
    pool1 = mpool_op(conv1_2,   name="pool1",   kh=2, kw=2, dw=2, dh=2)

    # block 2 -- outputs 56x56x128
    conv2_1 = tf.nn.dropout(conv_op(pool1,    name="conv2_1", kh=3, kw=3, n_out=32, dh=1, dw=1), dropout_keep_prob)
    conv2_2 = tf.nn.dropout(conv_op(conv2_1,  name="conv2_2", kh=3, kw=3, n_out=32, dh=1, dw=1), dropout_keep_prob)
    pool2 = mpool_op(conv2_2,   name="pool2",   kh=2, kw=2, dh=2, dw=2)

    # # block 3 -- outputs 28x28x256
    conv3_1 = tf.nn.dropout(conv_op(pool2,    name="conv3_1", kh=3, kw=3, n_out=64, dh=1, dw=1), dropout_keep_prob)
    conv3_2 = tf.nn.dropout(conv_op(conv3_1,  name="conv3_2", kh=3, kw=3, n_out=64, dh=1, dw=1), dropout_keep_prob)
    pool3 = mpool_op(conv3_2,   name="pool3",   kh=2, kw=2, dh=2, dw=2)

    # block 4 -- outputs 14x14x512
    conv4_1 = tf.nn.dropout(conv_op(pool3,    name="conv4_1", kh=3, kw=3, n_out=128, dh=1, dw=1), dropout_keep_prob)
    conv4_2 = tf.nn.dropout(conv_op(conv4_1,  name="conv4_2", kh=3, kw=3, n_out=128, dh=1, dw=1), dropout_keep_prob)
    conv4_3 = tf.nn.dropout(conv_op(conv4_2,  name="conv4_2", kh=3, kw=3, n_out=128, dh=1, dw=1), dropout_keep_prob)
    pool4 = mpool_op(conv4_3,   name="pool4",   kh=2, kw=2, dh=2, dw=2)


    # flatten
    shp = pool4.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool4, [-1, flattened_shape], name="resh1")

    # fully connected
    fc6 = fc_op(resh1, name="fc6", n_out=1024)
    fc6_drop = tf.nn.dropout(fc6, dropout_keep_prob, name="fc6_drop")

    fc7 = fc_op(fc6_drop, name="fc7", n_out=1024)
    fc7_drop = tf.nn.dropout(fc7, dropout_keep_prob, name="fc7_drop")

    fc8 = fc_op(fc7_drop, name="fc8", n_out=10)
    softmax = tf.nn.softmax(fc8)
    predictions = tf.argmax(softmax, 1)
    return predictions, softmax, fc8
