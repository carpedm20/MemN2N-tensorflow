import numpy as np
import tensorflow as tf

params = {
    'nwords': 10000,
    'edim': 150,
    'mem_size': 100,
    'batch_size': 125,
    'init_hid': 0.01,
}

#def g_build_model(params):
#input = tf.placeholder(tf.float32, [None, params['edim']])
#time = tf.placeholder(tf.float32, [None, params['mem_size']])

input = tf.Variable(np.ones([params['batch_size'], params['edim']]) * params['init_hid'])
target = tf.placeholder(tf.int32, [params['batch_size']])
context = tf.placeholder(tf.int32, [params['batch_size'], params['mem_size']])

# time
time_array = np.ndarray([params['batch_size'], params['mem_size']], dtype=np.int32)
for t in xrange(params['mem_size']):
    time_array[:,t].fill(t)
time = tf.constant(time_array)

#hid, share_list = build_memory(params, input, context, time)
z = tf.Variable(tf.random_normal(params['edim'], params['nwords']))

#def build_memory(params, input, context, time):
hid = []
hid.append(input)
share_list = []
share_list.append([])

A = tf.Variable(tf.random_uniform([params['nwords'], params['edim']], -0.1, 0.1))
B = tf.Variable(tf.random_uniform([params['nwords'], params['edim']], -0.1, 0.1))
C = tf.Variable(tf.random_uniform([params['nwords'], params['edim']], -0.1, 0.1))

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

for h in xrange(params['nhop']):
    hid3dim = tf.reshape(hid[0],[-1, 1, params['edim']])
#    MMaout = tf.matmul(hid3dim, Ain)
#    P = tf.softmax(Aout2dim)

