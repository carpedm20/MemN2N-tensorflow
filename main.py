import tensorflow as tf

from utils import pp
from data import read_data
from model import MemN2N

tf.app.flags.DEFINE_integer("edim", 150, "internal state dimension [150]")
tf.app.flags.DEFINE_integer("lindim", 75, "linear part of the state [75]")
tf.app.flags.DEFINE_integer("nhop", 6, "number of hops [6]")
tf.app.flags.DEFINE_integer("mem_size", 100, "memory size [100]")
tf.app.flags.DEFINE_integer("batch_size", 128, "batch size to use during training [128]")
tf.app.flags.DEFINE_integer("nepoch", 100, "number of epoch to use during training [100]")
tf.app.flags.DEFINE_float("init_lr", 0.01, "initial learning rate [0.01]")
tf.app.flags.DEFINE_float("init_hid", 0.1, "initial internal state value [0.1]")
tf.app.flags.DEFINE_float("init_std", 0.05, "weight initialization std [0.05]")
tf.app.flags.DEFINE_float("max_grad_norm", 50, "clip gradients to this norm [50]")
tf.app.flags.DEFINE_string("data_dir", "data", "data directory [data]")
tf.app.flags.DEFINE_string("data_name", "ptb", "data set name [ptb]")
tf.app.flags.DEFINE_boolean("show", False, "print progress.")

FLAGS = tf.app.flags.FLAGS

def main(_):
    count = []
    word2idx = {}

    train_data = read_data('%s/%s.train.txt' % (FLAGS.data_dir, FLAGS.data_name), count, word2idx)
    valid_data = read_data('%s/%s.valid.txt' % (FLAGS.data_dir, FLAGS.data_name), count, word2idx)
    test_data = read_data('%s/%s.test.txt' % (FLAGS.data_dir, FLAGS.data_name), count, word2idx)

    idx2word = dict(zip(word2idx.values(), word2idx.keys()))
    FLAGS.nwords = len(word2idx)

    pp.pprint(tf.app.flags.FLAGS.__flags)

    with tf.Session() as sess:
        model = MemN2N(FLAGS, sess)
        model.build_model()
        model.run(train_data, valid_data)

if __name__ == '__main__':
    tf.app.run()
