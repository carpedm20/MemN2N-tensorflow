import numpy as np
import tensorflow as tf

params = {
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

# Temporal Encoding
T_A = tf.Variable(tf.random_uniform([params['mem_size'], params['edim']], -0.1, 0.1))
T_B = tf.Variable(tf.random_uniform([params['mem_size'], params['edim']], -0.1, 0.1))

Ain_c = tf.nn.embedding_lookup(A, context)
Ain_t = tf.nn.embedding_lookup(T_A, time)
# m_i = sum A * x_ij + T_i
Ain = tf.add(Ain_c, Ain_t)

# Temporal Encoding
Bin_c = tf.nn.embedding_lookup(B, context)
Bin_t = tf.nn.embedding_lookup(T_B, time)
# u = sum B * x_ij + T_i
Bin = tf.add(Bin_c, Bin_t)

for h in xrange(1, params['nhop'] + 1):
    # [batch_size x mem_size x edim]
    hid3dim = tf.reshape(hid[h-1], [-1, 1, params['edim']])
    Aout = tf.batch_matmul(hid3dim, Ain, adj_y=True)
    # [batch_size x mem_size]
    Aout2dim = tf.reshape(Aout, [-1, params['mem_size']])

    # [batch_size x mem_size]
    P = tf.nn.softmax(Aout2dim)
    probs3dim = tf.reshape(Aout, [-1, 1, params['mem_size']])
    # [batch_size x mem_size]
    Bout = tf.batch_matmul(probs3dim, Bin, adj_y=True)

    # [batch_size x edim]
    Cout = tf.nn.linear(C, hid[h-1])
    D = tf.add(Cout, Bout)
    

#hid, share_list = build_memory(params, input, context, time)
z = tf.Variable(tf.random_normal(params['edim'], params['nwords']))
pred = tf.log(tf.softmax(z))
loss = -tf.reduce_sum(target * pred)

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
