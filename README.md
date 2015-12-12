End-To-End Memory Networks in Tensorflow
========================================

Tensorflow implementation of [End-To-End Memory Networks](http://arxiv.org/abs/1503.08895v4) for language modeling (see Section 5). The original torch code from Facebook can be found [here](https://github.com/facebook/MemNN/tree/master/MemN2N-lang-model).

![alt tag](http://i.imgur.com/nv89JLc.png)


Prerequisites
-------------

This code requires [Tensorflow](https://www.tensorflow.org/). There is a set of sample Penn Tree Bank (PTB) corpus in `data` directory, which is a popular benchmark for measuring quality of these models. But you can use your own text data set which should be formated like [this](data/).


Usage
-----

To train a model with 6 hops and memory size of 100, run the following command:

    $ python main.py --nhop 6 --mem_size 100

To see all training options, run:

    $ python main.py --help

which will print:

    usage: main.py [-h] [--edim EDIM] [--lindim LINDIM] [--nhop NHOP]
                  [--mem_size MEM_SIZE] [--batch_size BATCH_SIZE]
                  [--nepoch NEPOCH] [--init_lr INIT_LR] [--init_hid INIT_HID]
                  [--init_std INIT_STD] [--max_grad_norm MAX_GRAD_NORM]
                  [--data_dir DATA_DIR] [--data_name DATA_NAME] [--show SHOW]
                  [--noshow]

    optional arguments:
      -h, --help            show this help message and exit
      --edim EDIM           internal state dimension [150]
      --lindim LINDIM       linear part of the state [75]
      --nhop NHOP           number of hops [6]
      --mem_size MEM_SIZE   memory size [100]
      --batch_size BATCH_SIZE
                            batch size to use during training [128]
      --nepoch NEPOCH       number of epoch to use during training [100]
      --init_lr INIT_LR     initial learning rate [0.01]
      --init_hid INIT_HID   initial internal state value [0.1]
      --init_std INIT_STD   weight initialization std [0.05]
      --max_grad_norm MAX_GRAD_NORM
                            clip gradients to this norm [50]
      --data_dir DATA_DIR   data directory [data]
      --data_name DATA_NAME
                            data set name [ptb]
      --show SHOW           print progress [False]
      --noshow

(Optional) If you want to see a progress bar, install `progress` with `pip`:

    $ pip install progress
    $ python main.py --show True --nhop 6 --mem_size 100


Performance
-----------

The training output looks like:

    $ python main.py --nhop 6 --mem_size 100 --show True
    Read 929589 words from data/ptb.train.txt
    Read 73760 words from data/ptb.valid.txt
    Read 82430 words from data/ptb.test.txt
    {'batch_size': 128,
    'data_dir': 'data',
    'data_name': 'ptb',
    'edim': 150,
    'init_hid': 0.1,
    'init_lr': 0.01,
    'init_std': 0.05,
    'lindim': 75,
    'max_grad_norm': 50,
    'mem_size': 100,
    'nepoch': 100,
    'nhop': 6,
    'nwords': 10000,
    'show': True}
    I tensorflow/core/common_runtime/local_device.cc:25] Local device intra op parallelism threads: 12
    I tensorflow/core/common_runtime/direct_session.cc:45] Direct session inter op parallelism threads: 12
    Training |################################| 100.0% | ETA: 0s
    Testing |################################| 100.0% | ETA: 0s
    {'perplexity': 706.9750327516226, 'epoch': 0, 'valid_perplexity': 484.1358331457669, 'learning_rate': 0.01}
    Training |################################| 100.0% | ETA: 0s
    Testing |################################| 100.0% | ETA: 0s
    {'perplexity': 461.76683044676156, 'epoch': 1, 'valid_perplexity': 354.81370402184035, 'learning_rate': 0.01}
    Training |################################| 100.0% | ETA: 0s
    Testing |################################| 100.0% | ETA: 0s
    {'perplexity': 334.2083182652857, 'epoch': 2, 'valid_perplexity': 302.1831648697854, 'learning_rate': 0.01}
    Training |################################| 100.0% | ETA: 0s
    Testing |################################| 100.0% | ETA: 0s
    {'perplexity': 266.82651384396894, 'epoch': 3, 'valid_perplexity': 236.20763192385064, 'learning_rate': 0.01}
    Training |################################| 100.0% | ETA: 0s
    Testing |################################| 100.0% | ETA: 0s
    {'perplexity': 208.65395416960874, 'epoch': 4, 'valid_perplexity': 214.25900445147872, 'learning_rate': 0.01}
    Training |################################| 100.0% | ETA: 0s
    Testing |################################| 100.0% | ETA: 0s
    {'perplexity': 178.01065105989431, 'epoch': 5, 'valid_perplexity': 197.68565892455698, 'learning_rate': 0.01}
    Training |################################| 100.0% | ETA: 0s
    Testing |################################| 100.0% | ETA: 0s
    {'perplexity': 159.62503948293656, 'epoch': 6, 'valid_perplexity': 172.76667355642132, 'learning_rate': 0.01}
    Training |################################| 100.0% | ETA: 0s
    Testing |################################| 100.0% | ETA: 0s
    {'perplexity': 144.20704830477644, 'epoch': 7, 'valid_perplexity': 179.32751346077671, 'learning_rate': 0.01}
    Training |################################| 100.0% | ETA: 0s
    Testing |################################| 100.0% | ETA: 0s
    {'perplexity': 119.2060264104084, 'epoch': 8, 'valid_perplexity': 159.85036344063835, 'learning_rate': 0.006666666666666667}
    Training |################################| 100.0% | ETA: 0s
    Testing |################################| 100.0% | ETA: 0s
    {'perplexity': 110.78052987557061, 'epoch': 9, 'valid_perplexity': 154.4743895514825, 'learning_rate': 0.006666666666666667}
    Training |################################| 100.0% | ETA: 0s
    Testing |################################| 100.0% | ETA: 0s
    {'perplexity': 105.14122742021145, 'epoch': 10, 'valid_perplexity': 151.66942464731136, 'learning_rate': 0.006666666666666667}
    Training |##############                  | 44.0% | ETA: 378s

The performance comparison with the original paper will be upadated soon.


Author
------

Taehoon Kim / [@carpedm20](http://carpedm20.github.io/)
