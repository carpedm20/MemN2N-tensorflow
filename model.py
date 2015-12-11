import math
import numpy as np
import tensorflow as tf

class MemN2N(object):
    def __init__(self, config):
        self.nwords = config.get('nwords', 100000)
        self.nhop = config.get('nhop', 3)
        self.edim = config.get('edim', 125)
        self.init_hid = config.get('init_hid', 0.1)
        self.mem_size = config.get('mem_size', 150)
        self.batch_size = config.get('batch_size', 100)
        self.lindim = config.get('lindim', 75)

        self.input = tf.placeholder(tf.float32, [None, self.edim])
        self.time = tf.placeholder(tf.int32, [None, self.mem_size])
        self.target = tf.placeholder(tf.float32, [self.batch_size, self.nwords]) # should pass one-hot-encoded labels
        self.context = tf.placeholder(tf.int32, [self.batch_size, self.mem_size])

        self.hid = []
        self.hid.append(self.input)
        self.share_list = []
        self.share_list.append([])

        self.train = None
        self.loss = None

        self.build_memory()

    def build_memory(self):
        A = tf.Variable(tf.random_uniform([self.nwords, self.edim], -0.1, 0.1))
        B = tf.Variable(tf.random_uniform([self.nwords, self.edim], -0.1, 0.1))
        C = tf.Variable(tf.random_uniform([self.edim, self.edim], -0.1, 0.1))

        # Temporal Encoding
        T_A = tf.Variable(tf.random_uniform([self.mem_size, self.edim], -0.1, 0.1))
        T_B = tf.Variable(tf.random_uniform([self.mem_size, self.edim], -0.1, 0.1))

        # m_i = sum A_ij * x_ij + T_A_i
        Ain_c = tf.nn.embedding_lookup(A, self.context)
        Ain_t = tf.nn.embedding_lookup(T_A, self.time)
        Ain = tf.add(Ain_c, Ain_t)

        # c_i = sum B_ij * u + T_B_i
        Bin_c = tf.nn.embedding_lookup(B, self.context)
        Bin_t = tf.nn.embedding_lookup(T_B, self.time)
        Bin = tf.add(Bin_c, Bin_t)

        for h in xrange(self.nhop):
            self.hid3dim = tf.reshape(self.hid[-1], [-1, 1, self.edim])
            Aout = tf.batch_matmul(self.hid3dim, Ain, adj_y=True)
            Aout2dim = tf.reshape(Aout, [-1, self.mem_size])
            P = tf.nn.softmax(Aout2dim)

            probs3dim = tf.reshape(P, [-1, 1, self.mem_size])
            Bout = tf.batch_matmul(probs3dim, Bin)
            Bout2dim = tf.reshape(Bout, [-1, self.edim])

            Cout = tf.matmul(self.hid[-1], C)
            Dout = tf.add(Cout, Bout2dim)

            self.share_list[0].append(Cout)

            if self.lindim == self.edim:
                self.hid.append(Dout)
            elif self.lindim == 0:
                self.hid.append(tf.nn.relu(Dout))
            else:
                F = tf.slice(Dout, [0, 0], [self.batch_size, self.lindim])
                G = tf.slice(Dout, [0, self.lindim], [self.batch_size, self.edim-self.lindim])
                K = tf.nn.relu(G)
                self.hid.append(tf.concat(1, [F, K]))

    def build_model(self):
        z = tf.matmul(self.hid[-1], tf.Variable(tf.random_uniform([self.edim, self.nwords], -0.1, 0.1)))

        self.loss = tf.nn.softmax_cross_entropy_with_logits(z, self.target)
        self.train = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)

    def train(self, words):
        N = math.ceil(words.shape[1] / self.batch_size)
        cost = 0
        y = np.ones(1)

        x = np.ndarray([self.batch_size, self.mem_size], dtype=np.float32)
        x.fill(self.init_hid)

        time = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
        for idx in xrange(self.mem_size):
            time
