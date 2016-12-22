import numpy as np
import tensorflow as tf

def save_weights(graph, fpath):
    sess = tf.get_default_session()
    variables = graph.get_collection("variables")
    variable_names = [v.name for v in variables]
    kwargs = dict(zip(variable_names, sess.run(variables)))
    np.savez(fpath, **kwargs)

def load_weights(graph, fpath):
    sess = tf.get_default_session()
    variables = graph.get_collection("variables")
    data = np.load(fpath)
    for v in variables:
        if v.name not in data:
            print("could not load data for variable='%s'" % v.name)
            continue
        print("assigning %s" % v.name)
        sess.run(v.assign(data[v.name]))

def iterative_reduce(ops, inputs, args, batch_size, fn):
    """
    calls session.run for mini batches of batch_size in length

    Arguments:
        ops: tensor operations you want to call in sess.run
        inputs: a list of tensors you want to feed into feed_dict
        args: a list of arrays you want to split into minibatches and feed into feed_dict. This
              must be the same order as your inputs
        batch_size: size of your mini batch
        fn: aggregate each output from ops using this function (ex: lambda x: np.mean(x, axis=0))
    """
    sess = tf.get_default_session()
    N = len(args[0])
    results = []
    for i in range(0, N, batch_size):
        batch_start = i
        batch_end = i + batch_size
        minibatch_args = [a[batch_start:batch_end] for a in args]
        result = sess.run(ops, dict(zip(inputs, minibatch_args)))
        results.append(result)
    results = [fn(r) for r in zip(*results)]
    return results

class StatLogger:
    """
    file writer to record various statistics
    """

    def __init__(self, fpath):
        import os
        import os.path as pth

        self.fpath = fpath
        fdir = pth.split(fpath)[0]
        if len(fdir) > 0 and not pth.exists(fdir):
            os.makedirs(fdir)


    def report(self, step, **kwargs):
        import json
        with open(self.fpath, "a") as fh:
            data = {
                "step": step
            }
            data.update(kwargs)
            fh.write(json.dumps(data) + "\n")
