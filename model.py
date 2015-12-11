import math
import random
import numpy as np
import tensorflow as tf

class MemN2N(object):
    def __init__(self, config, sess):
        self.nwords = config.get('nwords', 100000)
        self.nhop = config.get('nhop', 3)
        self.edim = config.get('edim', 125)
        self.init_hid = config.get('init_hid', 0.1)
        self.mem_size = config.get('mem_size', 150)
        self.batch_size = config.get('batch_size', 100)
        self.lindim = config.get('lindim', 75)
        self.max_grad_norm = config.get('max_grad_norm', 50)
        self.show = config.get('show', False)

        self.input = tf.placeholder(tf.float32, [None, self.edim], name="input")
        self.time = tf.placeholder(tf.int32, [None, self.mem_size], name="time")
        self.target = tf.placeholder(tf.float32, [self.batch_size, self.nwords], name="target")
        self.context = tf.placeholder(tf.int32, [self.batch_size, self.mem_size], name="context")

        self.hid = []
        self.hid.append(self.input)
        self.share_list = []
        self.share_list.append([])

        self.lr = None
        self.loss = None
        self.step = None
        self.optim = None

        self.sess = sess
        self.log_loss = []
        self.log_perp = []

    def build_memory(self):
        self.global_step = tf.Variable(0, name="global_step")

        self.A = tf.Variable(tf.random_uniform([self.nwords, self.edim], -0.1, 0.1))
        self.B = tf.Variable(tf.random_uniform([self.nwords, self.edim], -0.1, 0.1))
        self.C = tf.Variable(tf.random_uniform([self.edim, self.edim], -0.1, 0.1))

        # Temporal Encoding
        self.T_A = tf.Variable(tf.random_uniform([self.mem_size, self.edim], -0.1, 0.1))
        self.T_B = tf.Variable(tf.random_uniform([self.mem_size, self.edim], -0.1, 0.1))

        # m_i = sum A_ij * x_ij + T_A_i
        Ain_c = tf.nn.embedding_lookup(self.A, self.context)
        Ain_t = tf.nn.embedding_lookup(self.T_A, self.time)
        Ain = tf.add(Ain_c, Ain_t)

        # c_i = sum B_ij * u + T_B_i
        Bin_c = tf.nn.embedding_lookup(self.B, self.context)
        Bin_t = tf.nn.embedding_lookup(self.T_B, self.time)
        Bin = tf.add(Bin_c, Bin_t)

        for h in xrange(self.nhop):
            self.hid3dim = tf.reshape(self.hid[-1], [-1, 1, self.edim])
            Aout = tf.batch_matmul(self.hid3dim, Ain, adj_y=True)
            Aout2dim = tf.reshape(Aout, [-1, self.mem_size])
            P = tf.nn.softmax(Aout2dim)

            probs3dim = tf.reshape(P, [-1, 1, self.mem_size])
            Bout = tf.batch_matmul(probs3dim, Bin)
            Bout2dim = tf.reshape(Bout, [-1, self.edim])

            Cout = tf.matmul(self.hid[-1], self.C)
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
        self.build_memory()

        self.W = tf.Variable(tf.random_uniform([self.edim, self.nwords], -0.1, 0.1))
        z = tf.matmul(self.hid[-1], self.W)

        self.lr = tf.train.exponential_decay(0.001, self.global_step,
                                              100, 0.96, staircase=True)
        self.loss = tf.nn.softmax_cross_entropy_with_logits(z, self.target)
        self.opt = tf.train.GradientDescentOptimizer(self.lr)

        params = [self.A, self.B, self.C, self.T_A, self.T_B, self.W]
        grads_and_vars = self.opt.compute_gradients(self.loss,params)
        clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], self.max_grad_norm), gv[1]) \
                                   for gv in grads_and_vars]

        inc = self.global_step.assign_add(1)
        with tf.control_dependencies([inc]):
            self.optim = self.opt.apply_gradients(clipped_grads_and_vars)

        tf.initialize_all_variables().run()
        self.saver = tf.train.Saver()

    def train(self, data):
        N = int(math.ceil(len(data) / self.batch_size))
        cost = 0

        x = np.ndarray([self.batch_size, self.edim], dtype=np.float32)
        time = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
        target = np.zeros([self.batch_size, self.nwords]) # one-hot-encoded
        context = np.ndarray([self.batch_size, self.mem_size])

        x.fill(self.init_hid)
        for t in xrange(self.mem_size):
            time[:,t].fill(t)

        if self.show:
            from utils import ProgressBar
            bar = ProgressBar('Training', max=N)

        for idx in xrange(N):
            if self.show: bar.next()
            for b in xrange(self.batch_size):
                m = random.randrange(self.mem_size, len(data))
                target[b][data[m]] = 1
                context[b] = data[m - self.mem_size:m]

            loss, self.step = self.sess.run([self.loss,
                                             self.global_step],
                                             feed_dict={
                                                 self.input: x,
                                                 self.time: time,
                                                 self.target: target,
                                                 self.context: context})
            cost += loss

        if self.show: bar.finish()
        return cost/N/self.batch_size

    def test(self, data):
        N = int(math.ceil(len(data) / self.batch_size))
        cost = 0

        x = np.ndarray([self.batch_size, self.edim], dtype=np.float32)
        time = np.ndarray([self.batch_size, self.mem_size], dtype=np.int32)
        target = np.zeros([self.batch_size, self.nwords]) # one-hot-encoded
        context = np.ndarray([self.batch_size, self.mem_size])

        x.fill(self.init_hid)
        for t in xrange(self.mem_size):
            time[:,t].fill(t)

        if self.show:
            from utils import ProgressBar
            bar = ProgressBar('Training', max=N)

        m = self.mem_size 
        for idx in xrange(N):
            if self.show: bar.next()
            for b in xrange(self.batch_size):
                target[b][data[m]] = 1
                context[b] = data[m - self.mem_size:m]
                m += 1

                if m >= len(data):
                    m = self.mem_size

            _, loss = self.sess.run([self.optim, self.loss], feed_dict={self.input: x,
                                                                        self.time: time,
                                                                        self.target: target,
                                                                        self.context: context})
            cost += loss

        if self.show: bar.finish()
        return cost/N/self.batch_size

    def run(self, train_data, test_data, epochs):
        for idx in xrange(epochs):
            train_loss = np.sum(self.train(train_data))
            test_loss = np.sum(self.test(test_data))

            self.log_loss.append([train_loss, test_loss])
            self.log_perp.append([math.exp(train_loss), math.ext(test_loss)])

            state = {
                'perplexity': math.exp(train_loss),
                'epoch': idx,
                'valid_perplexity': math.exp(test_loss)
            }
            print(state)

            if idx % 10 == 0:
                self.saver.save(self.sess,
                                "MemN2N.model",
                                 global_step = self.step.astype(int))
