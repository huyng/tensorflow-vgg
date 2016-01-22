# Adapted from https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/models/image/cifar10/cifar10.py

from datetime import datetime
import math
import time
import numpy as np
import dataset
import tensorflow.python.platform
import tensorflow as tf


batch_size = 8

def conv_op(input_op, name, kw, kh, n_out, dw, dh):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel_init_val = tf.truncated_normal([kh, kw, n_in, n_out], dtype=tf.float32, stddev=0.1)
        kernel = tf.Variable(kernel_init_val, trainable=True, name='w')
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        activation = tf.nn.relu(z, name=scope)
        return activation

def fc_op(input_op, name, n_out):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal([n_in, n_out], dtype=tf.float32, stddev=0.1), name='w')
        biases = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=tf.float32), name='b')
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=name)
        return activation

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


def inference_vgg(input_op, dropout_keep_prob):

    # assume input_op shape is 224x224x3

    # block 1 -- outputs 112x112x64
    conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1)
    conv1_2 = conv_op(conv1_1,  name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1)
    pool1 = mpool_op(conv1_2,   name="pool1",   kh=4, kw=4, dw=4, dh=4)

    # block 2 -- outputs 56x56x128
    conv2_1 = conv_op(pool1,    name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1)
    conv2_2 = conv_op(conv2_1,  name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1)
    pool2 = mpool_op(conv2_2,   name="pool2",   kh=4, kw=4, dh=4, dw=4)

    # # block 3 -- outputs 28x28x256
    # conv3_1 = conv_op(pool2,    name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1)
    # conv3_2 = conv_op(conv3_1,  name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1)
    # pool3 = mpool_op(conv3_2,   name="pool3",   kh=2, kw=2, dh=2, dw=2)
    #
    # # block 4 -- outputs 14x14x512
    # conv4_1 = conv_op(pool3,    name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1)
    # conv4_2 = conv_op(conv4_1,  name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1)
    # conv4_3 = conv_op(conv4_2,  name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1)
    # pool4 = mpool_op(conv4_3,   name="pool4",   kh=2, kw=2, dh=2, dw=2)
    #
    # # block 5 -- outputs 7x7x512
    # conv5_1 = conv_op(pool4,    name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1)
    # conv5_2 = conv_op(conv5_1,  name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1)
    # conv5_3 = conv_op(conv5_2,  name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1)
    # pool5 = mpool_op(conv5_3,   name="pool5",   kh=2, kw=2, dw=2, dh=2)

    # flatten
    shp = pool2.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool2, [-1, flattened_shape], name="resh1")

    # fully connected
    fc6 = fc_op(resh1, name="fc6", n_out=4096)
    fc6_drop = tf.nn.dropout(fc6, dropout_keep_prob, name="fc6_drop")

    fc7 = fc_op(fc6_drop, name="fc7", n_out=10)
    fc7_drop = tf.nn.dropout(fc7, dropout_keep_prob, name="fc7_drop")

    fc8 = fc_op(fc7_drop, name="fc8", n_out=10)
    return fc8


def random_test_input():
    """
    this generates random test input, useful for debugging
    """
    sz = 224
    channels = 3
    init_val = tf.random_normal(
        (batch_size, sz, sz, channels),
        dtype=tf.float32,
        stddev=1
    )
    images = tf.Variable(init_val)
    labels = tf.Variable(tf.ones([batch_size], dtype=tf.int32))
    return images, labels

def evaluate(predictions, labels):
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

def train(lr=0.0001, max_step=5000*10):
    """
    train model

    :param lr:  This is the learning rate
    """
    with tf.Graph().as_default():

        in_images = tf.placeholder("float", [batch_size, 32, 32, 3])
        images = tf.image.resize_images(in_images, 64, 64)
        labels = tf.placeholder("int32", [batch_size])
        dropout_keep_prob = tf.placeholder("float")


        # Build a Graph that computes the logits predictions from the
        # inference model.
        # last_layer = inference_vgg(images, dropout_keep_prob)
        last_layer = inference_vgg(images, dropout_keep_prob )

        # Add a simple objective so we can calculate the backward pass.
        objective = loss(last_layer, labels)
        _, total_correct = evaluate(last_layer, labels)
        optimizer = tf.train.RMSPropOptimizer(lr, 0.9)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_step = optimizer.minimize(objective, global_step=global_step)

        ema = tf.train.ExponentialMovingAverage(0.999)
        maintain_averages_op = ema.apply([objective])


        # grab summary variables we want to log
        tf.scalar_summary("loss function", objective)
        # tf.scalar_summary("accuracy", accuracy)
        tf.scalar_summary("avg loss function", ema.average(objective))

         # Create a saver.
        saver = tf.train.Saver(tf.all_variables())


        summary_op = tf.merge_all_summaries()

        # Build an initialization operation.
        initializer = tf.initialize_all_variables()

        # Start running operations on the Graph.
        with tf.Session() as sess:
            sess.run(initializer)
            writer = tf.train.SummaryWriter("train_logs", graph_def=sess.graph_def)
            trn, tst = dataset.get_cifar10(batch_size)
            for step in range(max_step):

                # get batch and format data
                batch = trn.next()
                X = np.vstack(batch[0]).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                Y = np.array(batch[1])

                t0 = time.time()
                result = sess.run(
                    [train_step, objective, summary_op, maintain_averages_op],
                    feed_dict = {
                        in_images: X,
                        labels: Y,
                        dropout_keep_prob: 0.5
                    }
                )
                duration = time.time() - t0

                if np.isnan(result[1]):
                    print("gradient vanished/exploded")
                    return

                if step % 10 == 0:
                    examples_per_sec = batch_size/duration
                    sec_per_batch = float(duration)
                    format_str = '%s: step %d, loss = %.4f (%.1f examples/sec; %.3f sec/batch)'
                    print(format_str % (datetime.now(), step, result[1], examples_per_sec, sec_per_batch))

                if step % 100 == 0:
                    writer.add_summary(result[2], step)

                if step % 1000 == 0:
                    print("%s: step %d, evaluating test set" % (datetime.now(), step))
                    correct_count = 0
                    num_tst_examples = tst[0].shape[0]
                    for tst_idx in range(0, num_tst_examples, batch_size):
                        X_tst = tst[0][tst_idx:np.min([tst_idx+batch_size, num_tst_examples]), :]
                        X_tst = X_tst.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                        Y_tst = tst[1][tst_idx:np.min([tst_idx+batch_size, num_tst_examples])]
                        correct_count += total_correct.eval({
                            in_images: X_tst,
                            labels: Y_tst,
                            dropout_keep_prob: 1.0
                        })
                    accuracy = float(correct_count)/num_tst_examples
                    print("%s tst accuracy = %.3f" % (datetime.now(), accuracy))
                    if accuracy > 0.9:
                        checkpoint_path = saver.save(sess, "checkpoints/model.ckpt")
                        print("saving model %s" % checkpoint_path)








if __name__ == '__main__':
    train()
