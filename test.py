#!/usr/bin/env python3

import os
import argparse
import random

import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from matplotlib import pyplot as plt

import model


def draw(x, x_, c, path):
    plt.figure()
    ax = plt.subplot(221)
    ax.matshow(np.reshape(x, [28, 28]), cmap='Greys')
    ax.axis('off')
    ax = plt.subplot(222)
    ax.matshow(np.reshape(x_, [28, 28]), cmap='Greys')
    ax.axis('off')
    ax = plt.subplot(212)
    ax.plot(c, 'o', ms=20, mfc='orange')
    ax.set_xticks(range(10))
    ax.set_xlabel('class')
    ax.set_ylabel('score')
    ax.grid()
    plt.savefig(path)
    plt.close()


argparser = argparse.ArgumentParser(description='MNIST Autocoder')
argparser.add_argument('--model', metavar='<model name>',
                       type=str, default='coder',
                       help='model to test')
argparser.add_argument('--draw',
                       action='store_true',
                       help='whether draw output')

args = argparser.parse_args()

model_path = os.path.join('model', args.model + '.ckpt')


x = tf.placeholder(tf.float32, [None, 28*28])
global_step = tf.Variable(0, name='global_step', trainable=False)

mnist = read_data_sets('tmp/MNIST_data')

with tf.Session() as sess:
    c, _ = model.classifier(x, 10)
    sc = tf.nn.sigmoid(c)
    x_, _ = model.decoder(c)
    loss = model.loss_decoder(x, x_)

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    print('"%s" loaded' % (model_path))

    eval_x_, eval_sc, eval_loss, step = sess.run([x_, sc, loss, global_step], feed_dict={x: mnist.test.images})
    print('loss: %g' % (eval_loss))

    if args.draw:
        dirpath = os.path.join('tmp', args.model, str(step))
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        for _ in range(64):
            i = random.randint(0, len(mnist.test.images))
            path = os.path.join(dirpath, str(i) + '.png')
            draw(mnist.test.images[i], eval_x_[i], eval_sc[i], path)
            print('test', i, 'saved to', path)
