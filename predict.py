import sys
import tensorflow as tf
import tfutils
import numpy as np
import vgg
from skimage.transform import resize
from skimage.io import imread

G = tf.Graph()
with G.as_default():
    images = tf.placeholder("float", [1, 224, 224, 3])
    logits = vgg.build(images, n_classes=10, training=False)
    probs = tf.nn.softmax(logits)

def predict(im):
    if im.shape != (224, 224, 3):
        im = resize(im, (224, 224))
    im = np.expand_dims(im, 0)
    sess = tf.get_default_session()
    return sess.run(probs, {images: im})

if __name__ == '__main__':
    im = imread(sys.argv[1])
    sess = tf.Session(graph=G)
    with sess.as_default():
        tfutils.load_weights(G, "weights.40.npz")
        print predict(im)
