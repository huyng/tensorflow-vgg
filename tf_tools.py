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

def run_iterative(ops, inputs, args, batch_size):
    """
    calls session.run for mini batches of batch_size in length

    Arguments:
        ops: tensor operations you want to call in sess.run
        inputs: a list of tensors you want to feed into feed_dict
        args: a list of arrays you want to split into minibatches and feed into feed_dict. This
              must be the same order as your inputs
        batch_size: size of your mini batch
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
    return results
