import vgg
import tensorflow as tf
import numpy as np
import time
import layers as L
import tfutils
import dataset
import matplotlib.pyplot as P

# ======================
# Training Configuration
# ======================

batch_size = 12
dset = dataset.get_cifar10(batch_size=batch_size)
trn_generator = dset[0]
valset = dset[1]
lr = 0.001
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
    objective = L.loss(logits, tf.one_hot(labels, n_classes))
    error_top5 = L.topK_error(probs, labels, K=5)
    error_top1 = L.topK_error(probs, labels, K=1)
    optimizer = tf.train.AdamOptimizer(lr)
    grad_step = optimizer.minimize(objective)
    init = tf.initialize_all_variables()


# initialize training session
# ===========================

sess = tf.Session(graph=G)
with sess.as_default():
    sess.run(init)

    # Start training loop
    i = 0
    for epoch in range(nb_epochs):
        for step in range(steps_per_epoch):
            batch_train = trn_generator.next()
            X_trn = np.array(batch_train[0])
            Y_trn = np.array(batch_train[1])
            P.ion()
            P.imshow(X_trn[0])
            P.show()

            ops = [grad_step, objective, error_top1, error_top5]
            inputs = {imraw: X_trn, labels: Y_trn}
            results = sess.run(ops, feed_dict=inputs)
            print("TRN step:%-5d error_top1: %.4f, error_top5: %.4f, loss:%s" % (i,  results[2], results[3], results[1]))

            # report evaluation metrics every 10 training steps
            if (i % 10) == 0:
                X_vld = valset[0][:40]
                Y_vld = valset[1][:40]
                inputs = [imraw, labels]
                args = [X_vld, Y_vld]
                ops = [error_top1, error_top5]
                results = tfutils.run_iterative(ops, inputs, args, batch_size=batch_size)
                results = np.mean(results, axis=0)
                print("VLD step:%-5d error_top1: %.4f, error_top5: %.4f" % (i, results[0], results[1]))
                tfutils.save_weights(G, "weights.%s" % i)
            i += 1
