import vgg as model
import tensorflow as tf
import numpy as np
import time


def train(trn_generator,
          val_generator,
          steps_per_epoch,
          lr=0.1,
          nb_epochs=10,
          batch_size=12,
          training_log_path="train_log.csv"):


    with tf.Graph().as_default():

        raw_images = tf.placeholder("float", [batch_size, 32, 32, 3])
        images = tf.image.resize_images(raw_images, 64, 64)
        labels = tf.placeholder("int32", [batch_size])
        predictions = model.inference_op(images, input_shape=64, training=True)
        objective = model.loss_op(predictions, labels, batch_size)
        accuracy, total_correct = model.evaluate_op(predictions, labels)
        optimizer = tf.train.GradientDescentOptimizer(lr)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_step = optimizer.minimize(objective, global_step=global_step)

        # Start running operations on the Graph.
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            training_log = open(training_log_path, "w")
            training_log.write("trn_loss,trn_acc\n")

            for epoch in range(nb_epochs):

                for step in range(steps_per_epoch):
                    # get batch and format data
                    batch = trn_generator.next()
                    X = np.array(batch[0])
                    Y = np.array(batch[1])

                    t0 = time.time()
                    result = sess.run(
                        [train_step, objective, accuracy],
                        feed_dict = {
                            raw_images: X,
                            labels: Y,
                        }
                    )
                    trn_loss = result[1]
                    trn_acc = result[2]
                    duration = time.time() - t0


                    # print debugging info
                    print("epoch:%5d, step:%5d, trn_loss: %s, trn_acc: %s," % (epoch, step, trn_loss, trn_acc))
                    training_log.write("%s,%s\n" % (trn_loss, trn_acc))



                    # if step % 10 == 0:
                    #     examples_per_sec = batch_size/duration
                    #     sec_per_batch = float(duration)
                    #     format_str = '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)'
                    #     print(format_str % (datetime.now(), step, result[1], examples_per_sec, sec_per_batch))
                    #
                    # if step % 1000 == 0:
                    #     print("%s: step %d, evaluating test set" % (datetime.now(), step))
                    #     correct_count = 0
                    #     num_tst_examples = tst[0].shape[0]
                    #     for tst_idx in range(0, num_tst_examples, batch_size):
                    #         X_tst = tst[0][tst_idx:np.min([tst_idx+batch_size, num_tst_examples]), :]
                    #         X_tst = X_tst.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                    #         Y_tst = tst[1][tst_idx:np.min([tst_idx+batch_size, num_tst_examples])]
                    #         correct_count += total_correct.eval({
                    #             raw_images: X_tst,
                    #             labels: Y_tst,
                    #             dropout_keep_prob: 1.0
                    #         })
                    #     print("%s tst accuracy is = %s" % (datetime.now(), float(correct_count)/num_tst_examples))

if __name__ == '__main__':
    import dataset
    batch_size = 12
    trn_generator, val_generator = dataset.get_cifar10(batch_size=batch_size)
    train(trn_generator, val_generator, steps_per_epoch=50000/batch_size, batch_size=batch_size)
