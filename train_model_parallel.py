import sys
import os
import os.path as pth
import time
import numpy as np
import tensorflow as tf
import json
import vgg
import layers as L
import dataset
import yaml
import tools
import argparse


# =====================================
# Training configuration default params
# =====================================
config = {}

# =========================
# customize your model here
# =========================
def build_model(input_data_tensor, input_label_tensor):
    num_classes = config["num_classes"]
    images = tf.image.resize_images(input_data_tensor, [224, 224])
    logits = vgg.build(images, n_classes=num_classes, training=True)
    probs = tf.nn.softmax(logits)
    loss = L.loss(logits, tf.one_hot(input_label_tensor, num_classes))
    error_top5 = L.topK_error(probs, input_label_tensor, K=5)
    error_top1 = L.topK_error(probs, input_label_tensor, K=1)

    # you must return a dictionary with at least the "loss" as a key
    return dict(loss=loss,
                logits=logits,
                error_top5=error_top5,
                error_top1=error_top1)


# =================================
#  generice multi-gpu training code
# =================================
def train(train_data_generator):
    checkpoint_dir = config["checkpoint_dir"]
    learning_rate = config['learning_rate']
    data_dims = config['data_dims']
    batch_size = config['batch_size']
    num_gpus = config['num_gpus']
    num_epochs = config['num_epochs']
    num_samples_per_epoch = config["num_samples_per_epoch"]
    pretrained_weights = config["pretrained_weights"]
    steps_per_epoch = num_samples_per_epoch // (batch_size * num_gpus)
    num_steps = steps_per_epoch * num_epochs
    checkpoint_iter = config["checkpoint_iter"]
    experiment_dir = config['experiment_dir']
    train_log_fpath = pth.join(experiment_dir, 'train.log')
    log = tools.MetricsLogger(train_log_fpath)


    # =====================
    # define training graph
    # =====================
    G = tf.Graph()
    with G.as_default(), tf.device('/cpu:0'):
        full_data_dims = [batch_size * num_gpus] + data_dims
        data = tf.placeholder(dtype=tf.float32,
                              shape=full_data_dims,
                              name='data')
        labels = tf.placeholder(dtype=tf.int32,
                                shape=[batch_size * num_gpus],
                                name='labels')

        # we split the large batch into sub-batches to be distributed onto each gpu
        split_data = tf.split(0, num_gpus, data)
        split_labels = tf.split(0, num_gpus, labels)

        # setup optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # setup one model replica per gpu to compute loss and gradient
        replica_grads = []
        for i in range(num_gpus):
            with tf.name_scope('tower_%d' % i), tf.device('/gpu:%d' % i):
                model = build_model(split_data[i], split_labels[i])
                loss = model["loss"]
                grads = optimizer.compute_gradients(loss)
                replica_grads.append(grads)
                tf.get_variable_scope().reuse_variables()

        # We must calculate the mean of each gradient. Note this is a
        # synchronization point across all towers.
        average_grad = L.average_gradients(replica_grads)
        grad_step = optimizer.apply_gradients(average_grad)
        train_step = tf.group(grad_step)
        init = tf.initialize_all_variables()

    # ==================
    # run training graph
    # ==================
    config_proto = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(graph=G, config=config_proto)
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)
    with sess.as_default():
        if pretrained_weights:
            print("-- loading weights from %s" % pretrained_weights)
            tools.load_weights(G, pretrained_weights)

        for step in range(num_steps):
            data_batch, label_batch = train_data_generator.next()
            inputs = {data: data_batch, labels: label_batch}
            results = sess.run([train_step, loss], inputs)
            print("step:%s loss:%s" % (step, results[1]))
            log.report(step=step, split="TRN", loss=float(results[1]))


            if (step % checkpoint_iter == 0) or (step + 1 == num_steps):
                print("-- saving check point")
                tools.save_weights(G, pth.join(checkpoint_dir, "weights.%s" % step))



def main(argv=None):
    num_gpus = config['num_gpus']
    batch_size = config['batch_size']
    checkpoint_dir = config["checkpoint_dir"]
    experiment_dir = config["experiment_dir"]

    # setup experiment and checkpoint directories
    if not pth.exists(experiment_dir):
        os.makedirs(experiment_dir)
    if not pth.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    train_data_generator, valset = dataset.get_cifar10(batch_size * num_gpus)
    train(train_data_generator)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='YAML formatted config file')
    args = parser.parse_args()
    with open(args.config_file) as fp:
        config.update(yaml.load(fp))

    print "Experiment config"
    print "------------------"
    print json.dumps(config, indent=4)
    print "------------------"
    main()
