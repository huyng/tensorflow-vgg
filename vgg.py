from datetime import datetime
import math
import time

import tensorflow.python.platform
import tensorflow as tf



batch_size = 2
parameters = []
conv_counter = 1
pool_counter = 1
affine_counter = 1


def layer_conv2d(inputs, n_in, n_out, kh, kw, dh, dw, padding):
    global conv_counter
    global parameters
    name = 'conv' + str(conv_counter)
    conv_counter += 1
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal([kh, kw, n_in, n_out], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(inputs, kernel, [1, dh, dw, 1], padding=padding)
        biases = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        conv1 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print conv1.name
        print biases.name
        return conv1

def layer_affine(inputs, n_in, n_out):
    global affine_counter
    global parameters
    name = 'affine' + str(affine_counter)
    affine_counter += 1
    with tf.name_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal([n_in, n_out], dtype=tf.float32, stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=tf.float32), trainable=True, name='biases')
        affine1 = tf.nn.relu_layer(inputs, kernel, biases, name=name)
        parameters += [kernel, biases]
        return affine1

def layer_maxpool(inputs, kh, kw, dh, dw):
    global pool_counter
    global parameters
    name = 'pool' + str(pool_counter)
    pool_counter += 1
    return tf.nn.max_pool(inputs,
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




def run_benchmark():
    global parameters
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
        conv1 = layer_conv2d(images, n_in=3, n_out=64, kh=3, kw=3, dh=1, dw=1, padding='SAME')
        pool1 = layer_maxpool(conv1, kh=2, kw=2, dh=2, dw=2)
        conv2 = layer_conv2d(pool1,  n_in=64, n_out=128, kh=3, kw=3, dh=1, dw=1, padding='SAME')
        pool2 = layer_maxpool(conv2, kh=2, kw=2, dh=2, dw=2)
        conv3 = layer_conv2d(pool2,  n_in=128, n_out=256, kh=3, kw=3, dh=1, dw=1, padding='SAME')
        conv4 = layer_conv2d(conv3,  n_in=256, n_out=256, kh=3, kw=3, dh=1, dw=1, padding='SAME')
        pool4 = layer_maxpool(conv4, kh=2, kw=2, dh=2, dw=2)
        conv5 = layer_conv2d(pool4,  n_in=256, n_out=512, kh=3, kw=3, dh=1, dw=1, padding='SAME')
        conv6 = layer_conv2d(conv5,  n_in=512, n_out=512, kh=3, kw=3, dh=1, dw=1, padding='SAME')
        pool6 = layer_maxpool(conv6, kh=2, kw=2, dh=2, dw=2)
        conv7 = layer_conv2d(pool6,  n_in=512, n_out=512, kh=3, kw=3, dh=1, dw=1, padding='SAME')
        conv8 = layer_conv2d(conv7,  n_in=512, n_out=512, kh=3, kw=3, dh=1, dw=1, padding='SAME')
        pool8 = layer_maxpool(conv8, kh=2, kw=2, dw=2, dh=2)
        resh1 = tf.reshape(pool8,    [-1, 512*7*7])
        affn1 = layer_affine(resh1,  n_in=512*7*7, n_out=4096)
        affn2 = layer_affine(affn1,  n_in=4096, n_out=4096)
        affn3 = layer_affine(affn2,  n_in=4096, n_out=1000)

        # Build an initialization operation.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        sess = tf.Session()
        sess.run(init)


        # Run the forward benchmark.
        sess.run(affn3)

        # Add a simple objective so we can calculate the backward pass.
        objective = loss(affn3, labels)
        # Compute the gradient with respect to all the parameters.
        # grad = tf.gradients(objective, parameters)
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(objective)
        for i in range(1000):
            print i
            sess.run(train_step)

        # Run the backward benchmark.
        # sess.run(grad)


def main(_):
  run_benchmark()


if __name__ == '__main__':
  tf.app.run()
