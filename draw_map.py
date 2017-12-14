#!/usr/bin/env python3

import os
import argparse
import random

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

import model


def draw(x, path, size):
    fig, axs = plt.subplots(nrows=size, ncols=size)
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0, hspace=0)
    for i in range(size):
        for j in range(size):
            axs[i][j].matshow(np.reshape(x[i * size + j], [28, 28]), cmap='Greys')
            axs[i][j].axis('off')
    plt.savefig(path, dpi=1024)
    plt.close()


argparser = argparse.ArgumentParser(description='MNIST Autocoder')
argparser.add_argument('--model', metavar='<model name>',
                       type=str, default='2d',
                       help='model to test')

args = argparser.parse_args()

model_path = os.path.join('model', args.model + '.ckpt')


c = tf.placeholder(tf.float32, [None, 2])
global_step = tf.Variable(0, name='global_step', trainable=False)


cs = []
size = 32
for i in range(size):
    for j in range(size):
        cs.append([i / (size-1), j / (size-1)])

with tf.Session() as sess:
    x_, _ = model.decoder(c)

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    print('"%s" loaded' % (model_path))

    eval_x_, step = sess.run([x_, global_step], feed_dict={c: cs})

    path = os.path.join('tmp', args.model)
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, str(step) + '.png')
    draw(eval_x_, path, size)
    print('map saved to', path)
