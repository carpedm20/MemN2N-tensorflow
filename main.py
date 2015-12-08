import tensorflow as tf

from data import g_read_words
from model import g_build_model

count = []
word2idx = {}
train_data = g_read_words('data/ptb.train.txt', count, word2idx)
valid_data = g_read_words('data/ptb.valid.txt', count, word2idx)
test_data = g_read_words('data/ptb.test.txt', count, word2idx)

idx2word = dict(zip(word2idx.values(), word2idx.keys()))

params = {
    'nhop': 3,
    'edim': 150,
    'lindim': 75,
    'mem_size': 100,
    'batch_size': 125,
    'init_hid': 0.01,
    'nwords': 10000,
}

with tf.Session() as sess:
    _, model = g_build_model(params)

    init = tf.initialize_all_variables()
    sess.run(init)

    tf.train.write_graph(sess.graph_def, '/tmp/MemN2N', 'graph.pbtxt')
