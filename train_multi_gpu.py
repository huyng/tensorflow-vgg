import sys
import os
import os.path as pth
import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import json
import vgg
import layers as L
import dataset
from argparse import ArgumentParser

CONFIG = {}

# =========================
# customize your model here
# =========================

def model(input_data_tensor, input_label_tensor, CONFIG=None):
    num_classes = CONFIG["num_classes"]
    images = tf.image.resize_images(input_data_tensor, [224, 224])
    logits = vgg.build(images, n_classes=num_classes, training=True)
    loss = L.loss(logits, tf.one_hot(input_label_tensor, num_classes))
    return logits, loss


def train(train_data_generator):
    num_gpus              = CONFIG['num_gpus']
    batch_size            = CONFIG['batch_size']
    learning_rate         = CONFIG['learning_rate']
    experiment_dir        = CONFIG['experiment_dir']
    num_epochs            = CONFIG['num_epochs']
    data_dims             = CONFIG['data_dims']
    num_samples_per_epoch = CONFIG["num_samples_per_epoch"]
    checkpoint_dir = pth.join(experiment_dir, 'checkpoints')


    # =====================
    # define training graph
    # =====================
    G = tf.Graph()
    with G.as_default(), tf.device('/cpu:0'):
        full_data_dims = [batch_size * num_gpus] + data_dims
        data = tf.placeholder(dtype=tf.float32, shape=full_data_dims, name='data')
        labels = tf.placeholder(dtype=tf.int32, shape=[batch_size * num_gpus], name='labels')

        # we split the large batch into sub-batches to be distributed onto each gpu
        split_data = tf.split(0, num_gpus, data)
        split_labels = tf.split(0, num_gpus, labels)

        # setup optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # setup one tower per gpu to compute loss and gradient
        tower_grads = []
        for i in range(num_gpus):
            with tf.name_scope('tower_%d' % i), tf.device('/gpu:%d' % i):
                logits, loss = model(split_data[i], split_labels[i], CONFIG=CONFIG)
                grads = optimizer.compute_gradients(loss)
                tower_grads.append(grads)
                tf.get_variable_scope().reuse_variables()

        # We must calculate the mean of each gradient. Note this is a synchronization point across all towers.
        average_grad = L.average_gradients(tower_grads)
        grad_step = optimizer.apply_gradients(average_grad)
        train_step = tf.group(grad_step)
        init = tf.initialize_all_variables()
        saver = tf.train.Saver(tf.all_variables())

    # ==================
    # run training graph
    # ==================
    config_proto = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(graph=G, config=config_proto)
    with sess.as_default():
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)
        num_batches_per_epoch = num_samples_per_epoch // (batch_size * num_gpus)
        for step in range(num_batches_per_epoch * num_epochs):
            data_batch, label_batch = train_data_generator.next()
            inputs = {data: data_batch, labels: label_batch}
            results = sess.run([train_step, loss], inputs)
            print("step:%s loss:%s" % (step, results[1]))

            # Save the model checkpoint after each epoch
            if (step > 0) and (step % num_batches_per_epoch == 0 or (step + 1) == num_batches_per_epoch * num_epochs):
                checkpoint_path = pth.join(checkpoint_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):
    num_gpus       = CONFIG['num_gpus']
    batch_size     = CONFIG['batch_size']
    experiment_dir = CONFIG['experiment_dir']

    # setup experiment and checkpoint directories
    checkpoint_dir  = pth.join(experiment_dir, 'checkpoints')
    if not pth.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    print("----------------------------------------------------\n")
    print("Experiment Configuration\n")
    print("----------------------------------------------------\n")
    for param, value in sorted(CONFIG.items()):
        print("%s:\t" % param)
        print(str(value))
        print("\n")
    print("----------------------------------------------------\n")

    train_data_generator, valset = dataset.get_cifar10(batch_size*num_gpus)
    train(train_data_generator)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config_file', help='JSON-formatted config file')
    args = parser.parse_args()
    with open(args.config_file) as fp:
        CONFIG.update(json.load(fp))
    main()
