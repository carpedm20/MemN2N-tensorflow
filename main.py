import tensorflow as tf

from data import g_read_words
from model import MemN2N

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

if __name__ == '__main__':
    with tf.Session() as sess:
        model = MemN2N(params, sess)
        model.build_model()
        model.run(train_data, test_data, 100)

        tf.train.write_graph(sess.graph_def, '/tmp/MemN2N', 'graph.pbtxt')
