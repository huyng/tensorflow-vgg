import vgg
import tensorflow as tf
import numpy as np
import time
import layers as L
import tf_tools
import dataset


# ======================
# Training Configuration
# ======================

batch_size = 64
dset = dataset.get_cifar10(batch_size=batch_size)
trn_generator = dset[0]
valset = dset[1]
lr = 0.000001
nb_epochs = 10
n_classes = 10
steps_per_epoch = 50000/batch_size

#################################################################

# construct training graph
# ========================

G = tf.Graph()
with G.as_default():
    imraw = tf.placeholder(tf.float32, [None, 32, 32, 3])
    images = tf.image.resize_images(imraw, [224, 224])
    labels = tf.placeholder("int32", [None])
    logits = vgg.build(images, n_classes=n_classes, training=True)
    probs = tf.nn.softmax(logits)
    loss = L.loss(logits, tf.one_hot(labels, n_classes))
    error_top5 = L.topK_error(probs, labels, K=5)
    error_top1 = L.topK_error(probs, labels, K=1)
    optimizer = tf.train.AdamOptimizer(lr)
    grads = optimizer.compute_gradients(loss)
    grad_step = optimizer.apply_gradients(grads)
    init = tf.global_variables_initializer()


# initialize training session
# ===========================
log = tf_tools.StatLogger("train.log")
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
sess = tf.Session(graph=G, config=config)
with sess.as_default():
    sess.run(init)

    # Start training loop
    for epoch in range(nb_epochs):
        for step in range(steps_per_epoch):
            iteration = epoch*steps_per_epoch + step
            batch_train = trn_generator.next()
            X_trn = np.array(batch_train[0])
            Y_trn = np.array(batch_train[1])

            ops = [grad_step, loss, error_top1, error_top5]
            inputs = {imraw: X_trn, labels: Y_trn}
            results = sess.run(ops, feed_dict=inputs)
            print("TRN step:%-5d error_top1: %.4f, error_top5: %.4f, loss:%s" % (iteration, results[2], results[3], results[1]))
            log.report(step=iteration, split="TRN", error_top5=float(results[2]), error_top1=float(results[3]), loss=float(results[1]))

            # report evaluation metrics every 10 training steps
            if step % 100 == 0:
                X_vld = valset[0][:1000]
                Y_vld = valset[1][:1000]
                inputs = [imraw, labels]
                args = [X_vld, Y_vld]
                ops = [error_top1, error_top5, loss]
                results = tf_tools.run_iterative(ops, inputs, args, batch_size=batch_size)
                results = np.mean(results, axis=0)
                print("VLD step:%-5d error_top1: %.4f, error_top5: %.4f, loss:%s" % (iteration, results[0], results[1], results[2]))
                log.report(step=iteration, split="VLD", error_top5=float(results[1]), error_top1=float(results[0]), loss=float(results[2]))

            if step % 100 == 0:
                print("saving check point")
                tf_tools.save_weights(G, "weights.%s" % iteration)
