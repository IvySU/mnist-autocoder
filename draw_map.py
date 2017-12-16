#!/usr/bin/env python3

import os
import argparse
import random

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

import model


def draw(x, path, size):
    fig, axs = plt.subplots(nrows=size, ncols=size, figsize=[4, 4])
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    for i in range(size):
        for j in range(size):
            axs[j][i].matshow(np.reshape(x[i * size + j], [28, 28]), cmap='Greys')
            axs[j][i].axis('off')
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
        cs.append([(i+0.5) / (size-1), (j+0.5) / (size-1)])

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
