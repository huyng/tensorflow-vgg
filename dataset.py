import itertools as it
import numpy as np


def get_cifar10(batch_size=16):
    from skdata.cifar10.dataset import CIFAR10
    cifar10 = CIFAR10()
    cifar10.fetch(True)

    trn_labels = []
    trn_pixels = []
    for i in range(1,6):
        data = cifar10.unpickle("data_batch_%d" % i)
        trn_pixels.append(data['data'])
        trn_labels.extend(data['labels'])

    trn_pixels = np.vstack(trn_pixels)

    tst_data = cifar10.unpickle("test_batch")
    tst_labels = tst_data["labels"]
    tst_pixels = tst_data["data"]

    trn_set = batch_iterator(zip(trn_pixels, trn_labels), batch_size, batch_fn=lambda x: zip(*x))
    tst_set = batch_iterator(zip(tst_pixels, tst_labels), batch_size, batch_fn=lambda x: zip(*x))

    return trn_set, tst_set

def batch_iterator(iterable, size, cycle=False, batch_fn=lambda x: x):
    """
    Iterate over a list or iterator in batches
    """
    batch = []

    # loop to begining upon reaching end of iterable, if cycle flag is set
    if cycle is True:
        iterable = it.cycle(iterable)

    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch_fn(batch)
            batch = []

    if len(batch) > 0:
        yield batch_fn(batch)





if __name__ == '__main__':
    trn, tst = get_cifar10()
