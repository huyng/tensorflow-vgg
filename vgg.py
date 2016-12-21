from datetime import datetime
import math
import time
import numpy as np
import dataset
import tensorflow.python.platform
import tensorflow as tf
import layers as L


def build(input_tensor, n_classes=1000, rgb_mean=None, training=True):
    # assuming 224x224x3 input_tensor

    # define image mean
    if rgb_mean is None:
        rgb_mean = np.array([116.779, 123.68, 103.939], dtype=np.float32)
    mu = tf.constant(rgb_mean, name="rgb_mean")
    keep_prob = 0.5

    # subtract image mean
    net = tf.sub(input_tensor, mu, name="input_mean_centered")

    # block 1 -- outputs 112x112x64
    net = L.conv(net, name="conv1_1", kh=3, kw=3, n_out=64)
    net = L.conv(net, name="conv1_2", kh=3, kw=3, n_out=64)
    net = L.pool(net, name="pool1", kh=2, kw=2, dw=2, dh=2)

    # block 2 -- outputs 56x56x128
    net = L.conv(net, name="conv2_1", kh=3, kw=3, n_out=128)
    net = L.conv(net, name="conv2_2", kh=3, kw=3, n_out=128)
    net = L.pool(net, name="pool2", kh=2, kw=2, dh=2, dw=2)

    # # block 3 -- outputs 28x28x256
    net = L.conv(net, name="conv3_1", kh=3, kw=3, n_out=256)
    net = L.conv(net, name="conv3_2", kh=3, kw=3, n_out=256)
    net = L.pool(net, name="pool3", kh=2, kw=2, dh=2, dw=2)

    # block 4 -- outputs 14x14x512
    net = L.conv(net, name="conv4_1", kh=3, kw=3, n_out=512)
    net = L.conv(net, name="conv4_2", kh=3, kw=3, n_out=512)
    net = L.conv(net, name="conv4_3", kh=3, kw=3, n_out=512)
    net = L.pool(net, name="pool4", kh=2, kw=2, dh=2, dw=2)

    # block 5 -- outputs 7x7x512
    net = L.conv(net, name="conv5_1", kh=3, kw=3, n_out=512)
    net = L.conv(net, name="conv5_2", kh=3, kw=3, n_out=512)
    net = L.conv(net, name="conv5_3", kh=3, kw=3, n_out=512)
    net = L.pool(net, name="pool5", kh=2, kw=2, dw=2, dh=2)

    # flatten
    flattened_shape = np.prod([s.value for s in net.get_shape()[1:]])
    net = tf.reshape(net, [-1, flattened_shape], name="flatten")

    # fully connected
    net = L.fully_connected(net, name="fc6", n_out=4096)
    net = tf.nn.dropout(net, keep_prob)
    net = L.fully_connected(net, name="fc7", n_out=4096)
    net = tf.nn.dropout(net, keep_prob)
    net = L.fully_connected(net, name="fc8", n_out=n_classes)
    return net

if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [10, 224, 224, 3])
    net = build(x)
