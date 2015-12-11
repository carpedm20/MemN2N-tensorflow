import tensorflow as tf

from data import read_data
from model import MemN2N

tf.app.flags.DEFINE_integer("edim", 150, "internal state dimension.")
tf.app.flags.DEFINE_integer("lindim", 75, "linear part of the state.")
tf.app.flags.DEFINE_integer("nhop", 6, "number of hops.")
tf.app.flags.DEFINE_integer("mem_size", 100, "memory size.")
tf.app.flags.DEFINE_integer("batch_size", 125, "batch size to use during training.")
tf.app.flags.DEFINE_integer("nepoch", 100, "number of epoch to use during training.")
tf.app.flags.DEFINE_float("init_lr", 0.01, "initial learning rate.")
tf.app.flags.DEFINE_float("init_hid", 0.1, "initial internal state value.")
tf.app.flags.DEFINE_float("init_std", 0.05, "weight initialization std.")
tf.app.flags.DEFINE_float("max_grad_norm", 50, "clip gradients to this norm.")
tf.app.flags.DEFINE_string("data_dir", "data", "data directory.")
tf.app.flags.DEFINE_boolean("show", False, "print progress.")

FLAGS = tf.app.flags.FLAGS

def main(_):
    count = []
    word2idx = {}

    train_data = read_data('data/ptb.train.txt', count, word2idx)
    valid_data = read_data('data/ptb.valid.txt', count, word2idx)
    test_data = read_data('data/ptb.test.txt', count, word2idx)

    idx2word = dict(zip(word2idx.values(), word2idx.keys()))
    FLAGS.nwords = len(word2idx)

    with tf.Session() as sess:
        model = MemN2N(FLAGS, sess)
        model.build_model()
        model.run(train_data, test_data)

if __name__ == '__main__':
    tf.app.run()
