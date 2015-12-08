import numpy as np
import tensorflow as tf

params = {
    'pe': True,
    'nhop': 3,
    'edim': 150,
    'mem_size': 100,
    'batch_size': 125,
    'init_hid': 0.01,
    'nwords': 10000,
}

#def g_build_model(params):
#input = tf.placeholder(tf.float32, [None, params['edim']])
#time = tf.placeholder(tf.float32, [None, params['mem_size']])

# [batch_size x edim]
input = tf.Variable(np.ones([params['batch_size'], params['edim']], dtype=np.float32) * params['init_hid'])
target = tf.placeholder(tf.int32, [params['batch_size']])

# ???
# [batch_size x mem_size]
context = tf.placeholder(tf.int32, [params['batch_size'], params['mem_size']])

# time
time_array = np.ndarray([params['batch_size'], params['mem_size']], dtype=np.int32)
for t in xrange(params['mem_size']):
    time_array[:,t].fill(t)
# [batch_size x mem_size]
time = tf.constant(time_array)

#def build_memory(params, input, context, time):
hid = []
hid.append(input)
share_list = []
share_list.append([])

A = tf.Variable(tf.random_uniform([params['nwords'], params['edim']], -0.1, 0.1))
B = tf.Variable(tf.random_uniform([params['nwords'], params['edim']], -0.1, 0.1))
C = tf.Variable(tf.random_uniform([params['edim'], params['edim']], -0.1, 0.1))

# Position Encoding
if params['pe'] == True:
    PE_A = tf.Variable(tf..)

# Temporal Encoding
T_A = tf.Variable(tf.random_uniform([params['mem_size'], params['edim']], -0.1, 0.1))
T_B = tf.Variable(tf.random_uniform([params['mem_size'], params['edim']], -0.1, 0.1))

# [batch_size x mem_size x edim]
Ain_c = tf.nn.embedding_lookup(A, context)
Ain_t = tf.nn.embedding_lookup(T_A, time)
# m_i = sum A_ij * x_ij + T_i
# [batch_size x mem_size x edim]
Ain = tf.add(Ain_c, Ain_t)

# Temporal Encoding
Bin_c = tf.nn.embedding_lookup(B, context)
Bin_t = tf.nn.embedding_lookup(T_B, time)
# u = sum B * x_ij + T_i
# [batch_size x mem_size x edim]
Bin = tf.add(Bin_c, Bin_t)

for h in xrange(1, params['nhop'] + 1):
    # [batch_size x 1 x edim]
    hid3dim = tf.reshape(hid[h-1], [-1, 1, params['edim']])
    # [batch_size x 1 x edim]
    Aout = tf.batch_matmul(hid3dim, Ain, adj_y=True)
    # [batch_size x mem_size] : u.T * m_i
    Aout2dim = tf.reshape(Aout, [-1, params['mem_size']])
    # [batch_size x mem_size] : softmax(u.T * m_i), p is a probability vector over the inputs.
    P = tf.nn.softmax(Aout2dim)

    probs3dim = tf.reshape(P, [-1, 1, params['mem_size']])
    # o = sum_i p_i * b_i
    # [batch_size x edim]
    Bout = tf.batch_matmul(probs3dim, Bin)

    # shareList?
    # [batch_size x edim]
    Cout = tf.matmul(hid[h-1], C)
    Dout = tf.add(Cout, Bout)

    if params['lindim'] == params['edim']:
        hid[h] = Dout
    elif params['lindim'] == 0
        hid[h] = tf.nn.relu(Dout)
    else:
        F = tf.slice(Dout, [0, 0], [params['batch_size'], params['lindim'])
        G = tf.slice(Dout, [0, params['lindim']], [params['batch_size'], params['edim']-params['lindim']])
        K = tf.nn.relu(G)
        hid[h] = tf.concat(1, [F, K])

#hid, share_list = build_memory(params, input, context, time)
z = tf.Variable(tf.random_normal(params['edim'], params['nwords']))
pred = tf.log(tf.softmax(z))
loss = -tf.reduce_sum(target * pred)

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
