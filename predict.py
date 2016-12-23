import sys
import tensorflow as tf
import tools
import numpy as np
import vgg
import argparse
from skimage.transform import resize
from skimage.io import imread

G = tf.Graph()
with G.as_default():
    images = tf.placeholder("float", [1, 224, 224, 3])
    logits = vgg.build(images, n_classes=10, training=False)
    probs = tf.nn.softmax(logits)

def predict(im):
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']
    if im.shape != (224, 224, 3):
        im = resize(im, (224, 224))
    im = np.expand_dims(im, 0)
    sess = tf.get_default_session()
    results = sess.run(probs, {images: im})
    return labels[np.argmax(results)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", required=True, help="path to weights.npz file")
    parser.add_argument("image", help="path to jpg image")
    args = parser.parse_args()
    im = imread(args.image)
    sess = tf.Session(graph=G)
    with sess.as_default():
        tools.load_weights(G, args.weights)
        print predict(im)
