import numpy as np
import tensorflow as tf

def build_memory(params, input, context, time):
    hid = []
    hid.append(input)
    share_list = []
    share_list.append([])

    A = tf.Variable(tf.random_uniform([params['nwords'], params['edim']], -0.1, 0.1))
    B = tf.Variable(tf.random_uniform([params['nwords'], params['edim']], -0.1, 0.1))
    C = tf.Variable(tf.random_uniform([params['edim'], params['edim']], -0.1, 0.1))

    # Temporal Encoding
    T_A = tf.Variable(tf.random_uniform([params['mem_size'], params['edim']], -0.1, 0.1))
    T_B = tf.Variable(tf.random_uniform([params['mem_size'], params['edim']], -0.1, 0.1))

    # m_i = sum A_ij * x_ij + T_A_i
    Ain_c = tf.nn.embedding_lookup(A, context)
    Ain_t = tf.nn.embedding_lookup(T_A, time)
    Ain = tf.add(Ain_c, Ain_t)

    # c_i = sum B_ij * u + T_B_i
    Bin_c = tf.nn.embedding_lookup(B, context)
    Bin_t = tf.nn.embedding_lookup(T_B, time)
    Bin = tf.add(Bin_c, Bin_t)

    for h in xrange(params['nhop']):
        hid3dim = tf.reshape(hid[-1], [-1, 1, params['edim']])
        Aout = tf.batch_matmul(hid3dim, Ain, adj_y=True)
        Aout2dim = tf.reshape(Aout, [-1, params['mem_size']])
        P = tf.nn.softmax(Aout2dim)

        probs3dim = tf.reshape(P, [-1, 1, params['mem_size']])
        Bout = tf.batch_matmul(probs3dim, Bin)
        Bout2dim = tf.reshape(Bout, [-1, params['edim']])

        Cout = tf.matmul(hid[-1], C)
        Dout = tf.add(Cout, Bout2dim)

        share_list[0].append(Cout)

        if params['lindim'] == params['edim']:
            hid.append(Dout)
        elif params['lindim'] == 0:
            hid.append(tf.nn.relu(Dout))
        else:
            F = tf.slice(Dout, [0, 0], [params['batch_size'], params['lindim']])
            G = tf.slice(Dout, [0, params['lindim']], [params['batch_size'], params['edim']-params['lindim']])
            K = tf.nn.relu(G)
            hid.append(tf.concat(1, [F, K]))

    return hid, share_list

def g_build_model(params):
    input = tf.placeholder(tf.float32, [None, params['edim']])
    time = tf.placeholder(tf.int32, [None, params['mem_size']])
    # should pass one-hot-encoded labels
    target = tf.placeholder(tf.float32, [params['batch_size'], params['nwords']])
    context = tf.placeholder(tf.int32, [params['batch_size'], params['mem_size']])

    hid, share_list = build_memory(params, input, context, time)
    z = tf.matmul(hid[-1], tf.Variable(tf.random_uniform([params['edim'], params['nwords']], -0.1, 0.1)))

    # Negative Log Likelihood
    pred = tf.log(tf.nn.softmax(z))
    loss = -tf.reduce_mean(pred * target)

    train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    return None, loss
