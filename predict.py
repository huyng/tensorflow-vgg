from vgg import inference_vgg, batch_size
import tensorflow as tf
import numpy as np

def predict(img):
    X = np.zeros((batch_size, 32,32, 3))
    X[0] = img
    with tf.Graph().as_default():
        in_images = tf.placeholder("float", [batch_size, 32, 32, 3])
        images = tf.image.resize_images(in_images, 64, 64)
        inference_op = inference_vgg(images, dropout_keep_prob=tf.constant(1.0, dtype=tf.float32), input_shape=64)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, "checkpoints/model.ckpt")
            Y = sess.run(inference_op, feed_dict={in_images: X})
            return Y[0]

if __name__ == '__main__':
    import dataset
    trn, tst = dataset.get_cifar10(10)
    d = tst[11][1].reshape(3,32,32).transpose(1,2,0)
    print predict(d)
