from datetime import datetime
import math
import time
import numpy as np
import dataset
import tensorflow.python.platform
import tensorflow as tf


def conv(input_op, name, kw, kh, n_out, dw=1, dh=1):
    n_in = input_op.get_shape()[-1].value
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', [kh, kw, n_in, n_out], tf.float32, tf.truncated_normal_initializer())
        biases = tf.get_variable("bias", [n_out], tf.float32, tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input_op, weights, (1, dh, dw, 1), padding='SAME')
        activation = tf.nn.relu(tf.nn.bias_add(conv, biases))
        return activation


def fully_connected(input_op, name, n_out, activation_fn=tf.nn.relu):
    n_in = input_op.get_shape()[-1].value
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', [n_in, n_out], tf.float32, tf.truncated_normal_initializer())
        biases = tf.get_variable("bias", [n_out], tf.float32, tf.constant_initializer(0.0))
        logits = tf.nn.bias_add(tf.matmul(input_op, weights), biases)
        return activation_fn(logits)


def pool(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='VALID',
                          name=name)


def loss(logits, onehot_labels):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits, onehot_labels, name='xentropy')
    loss = tf.reduce_mean(xentropy, name='loss')
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
