from datetime import datetime
import math
import time
import numpy as np
import dataset
import tensorflow.python.platform
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer 

def conv(input_tensor, name, kw, kh, n_out, dw=1, dh=1, activation_fn=tf.nn.relu):
    n_in = input_tensor.get_shape()[-1].value
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', [kh, kw, n_in, n_out], tf.float32, xavier_initializer())
        biases = tf.get_variable("bias", [n_out], tf.float32, tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input_tensor, weights, (1, dh, dw, 1), padding='SAME')
        activation = activation_fn(tf.nn.bias_add(conv, biases))
        return activation


def fully_connected(input_tensor, name, n_out, activation_fn=tf.nn.relu):
    n_in = input_tensor.get_shape()[-1].value
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', [n_in, n_out], tf.float32, xavier_initializer())
        biases = tf.get_variable("bias", [n_out], tf.float32, tf.constant_initializer(0.0))
        logits = tf.nn.bias_add(tf.matmul(input_tensor, weights), biases)
        return activation_fn(logits)


def pool(input_tensor, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_tensor,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='VALID',
                          name=name)


def loss(logits, onehot_labels):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits, onehot_labels, name='xentropy')
    loss = tf.reduce_mean(xentropy, name='loss')
    return loss


def topK_error(predictions, labels, K=5):
    correct = tf.cast(tf.nn.in_top_k(predictions, labels, K), tf.float32)
    accuracy = tf.reduce_mean(correct)
    error = 1.0 - accuracy
    return error
