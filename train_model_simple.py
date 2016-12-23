import json
import sys
import os
import os.path as pth
import vgg
import tensorflow as tf
import numpy as np
import time
import layers as L
import tools
import dataset
import yaml
import argparse


# =====================================
# Training configuration default params
# =====================================
config = {}

#################################################################

# customize your model here
# =========================
def build_model(input_data_tensor, input_label_tensor):
    num_classes = config["num_classes"]
    weight_decay = config["weight_decay"]
    images = tf.image.resize_images(input_data_tensor, [224, 224])
    logits = vgg.build(images, n_classes=num_classes, training=True)
    probs = tf.nn.softmax(logits)
    loss_classify = L.loss(logits, tf.one_hot(input_label_tensor, num_classes))
    loss_weight_decay = tf.reduce_sum(tf.pack([tf.nn.l2_loss(i) for i in tf.get_collection('variables')]))
    loss = loss_classify + weight_decay*loss_weight_decay
    error_top5 = L.topK_error(probs, input_label_tensor, K=5)
    error_top1 = L.topK_error(probs, input_label_tensor, K=1)

    # you must return a dictionary with loss as a key, other variables
    return dict(loss=loss,
                probs=probs,
                logits=logits,
                error_top5=error_top5,
                error_top1=error_top1)


def train(trn_data_generator, vld_data=None):
    learning_rate = config['learning_rate']
    experiment_dir = config['experiment_dir']
    data_dims = config['data_dims']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    num_samples_per_epoch = config["num_samples_per_epoch"]
    steps_per_epoch = num_samples_per_epoch // batch_size
    num_steps = steps_per_epoch * num_epochs
    checkpoint_dir = pth.join(experiment_dir, 'checkpoints')
    train_log_fpath = pth.join(experiment_dir, 'train.log')
    vld_iter = config["vld_iter"]
    checkpoint_iter = config["checkpoint_iter"]
    pretrained_weights = config.get("pretrained_weights", None)

    # ========================
    # construct training graph
    # ========================
    G = tf.Graph()
    with G.as_default():
        input_data_tensor = tf.placeholder(tf.float32, [None] + data_dims)
        input_label_tensor = tf.placeholder(tf.int32, [None])
        model = build_model(input_data_tensor, input_label_tensor)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads = optimizer.compute_gradients(model["loss"])
        grad_step = optimizer.apply_gradients(grads)
        init = tf.initialize_all_variables()


    # ===================================
    # initialize and run training session
    # ===================================
    log = tools.MetricsLogger(train_log_fpath)
    config_proto = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(graph=G, config=config_proto)
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)
    with sess.as_default():
        if pretrained_weights:
            print("-- loading weights from %s" % pretrained_weights)
            tools.load_weights(G, pretrained_weights)


        # Start training loop
        for step in range(num_steps):
            batch_train = trn_data_generator.next()
            X_trn = np.array(batch_train[0])
            Y_trn = np.array(batch_train[1])

            ops = [grad_step] + [model[k] for k in sorted(model.keys())]
            inputs = {input_data_tensor: X_trn, input_label_tensor: Y_trn}
            results = sess.run(ops, feed_dict=inputs)
            results = dict(zip(sorted(model.keys()), results[1:]))
            print("TRN step:%-5d error_top1: %.4f, error_top5: %.4f, loss:%s" % (step,
                                                                                 results["error_top1"],
                                                                                 results["error_top5"],
                                                                                 results["loss"]))
            log.report(step=step,
                       split="TRN",
                       error_top5=float(results["error_top5"]),
                       error_top1=float(results["error_top5"]),
                       loss=float(results["loss"]))

            # report evaluation metrics every 10 training steps
            if (step % vld_iter == 0):
                print("-- running evaluation on vld split")
                X_vld = vld_data[0]
                Y_vld = vld_data[1]
                inputs = [input_data_tensor, input_label_tensor]
                args = [X_vld, Y_vld]
                ops = [model[k] for k in sorted(model.keys())]
                results = tools.iterative_reduce(ops, inputs, args, batch_size=1, fn=lambda x: np.mean(x, axis=0))
                results = dict(zip(sorted(model.keys()), results))
                print("VLD step:%-5d error_top1: %.4f, error_top5: %.4f, loss:%s" % (step,
                                                                                     results["error_top1"],
                                                                                     results["error_top5"],
                                                                                     results["loss"]))
                log.report(step=step,
                           split="VLD",
                           error_top5=float(results["error_top5"]),
                           error_top1=float(results["error_top1"]),
                           loss=float(results["loss"]))

            if (step % checkpoint_iter == 0) or (step + 1 == num_steps):
                print("-- saving check point")
                tools.save_weights(G, pth.join(checkpoint_dir, "weights.%s" % step))

def main():
    batch_size = config['batch_size']
    experiment_dir = config['experiment_dir']

    # setup experiment and checkpoint directories
    checkpoint_dir = pth.join(experiment_dir, 'checkpoints')
    if not pth.exists(experiment_dir):
        os.makedirs(experiment_dir)
    if not pth.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    trn_data_generator, vld_data = dataset.get_cifar10(batch_size)
    train(trn_data_generator, vld_data)


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
