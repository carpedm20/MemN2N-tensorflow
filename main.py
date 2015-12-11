import tensorflow as tf

from data import read_data
from model import MemN2N

count = []
word2idx = {}

train_data = read_data('data/ptb.train.txt', count, word2idx)
valid_data = read_data('data/ptb.valid.txt', count, word2idx)
test_data = read_data('data/ptb.test.txt', count, word2idx)

idx2word = dict(zip(word2idx.values(), word2idx.keys()))

params = {
    'show': True,
    'nhop': 3,
    'edim': 150,
    'lindim': 75,
    'mem_size': 100,
    'batch_size': 125,
    'max_grad_norm': 50,
    'init_hid': 0.01,
    'nwords': 10000,
}

if __name__ == '__main__':
    with tf.Session() as sess:
        model = MemN2N(params, sess)
        model.build_model()
        model.run(train_data, test_data, 100)

        tf.train.write_graph(sess.graph_def, '/tmp/MemN2N', 'graph.pbtxt')
