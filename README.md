End-To-End Memory Networks in Tensorflow
========================================

Tensorflow implementation of [End-To-End Memory Networks](http://arxiv.org/abs/1503.08895v4) for language modeling (see Section 5 of the paper). The original torch code and dataset can be found [here](https://github.com/facebook/MemNN/).

![alt tag](http://i.imgur.com/nv89JLc.png)


Setup
-----

This code requires [Tensorflow](https://www.tensorflow.org/). Also, it uses CUDA to run on GPU for faster training. To train on Penn Treebank corpus, you should download it separately (should be formatted like [this](http://github.com/wojzaremba/lstm/tree/master/data)) and put in data subdirectory.


Usage
-----

To train a model with 6 hops and memory size of 100 (best model described in the paper), run the following command:

    $ python main.py --nhop 6 --memsize 100

To see all training options, run:

    $ python main.lua --help

(Optional) If you want to see a progress bar, install `progress` with `pip`:

    $ pip install progress
    $ python main.py --show --nhop 6 --memsize 100


Author
------

Taehoon Kim / [@carpedm20](http://carpedm20.github.io/)
