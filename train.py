#!/usr/bin/env python3

import os
import argparse

import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

import model


argparser = argparse.ArgumentParser(description='MNIST Autocoder')
argparser.add_argument('--restore', metavar='<model name>',
                       type=str, default=None,
                       help='model to restore')
argparser.add_argument('--saveto', metavar='<model name>',
                       type=str, default=None,
                       help='model name to be saved as')

args = argparser.parse_args()

if args.restore:
    restore_model_path = os.path.join('model', args.restore + '.ckpt')
else:
    restore_model_path = None
if args.saveto:
    saveto_model_path = os.path.join('model', args.saveto + '.ckpt')
else:
    saveto_model_path = None


x = tf.placeholder(tf.float32, [None, 28*28])

global_step = tf.Variable(0, name='global_step', trainable=False)
inc_global_step = global_step.assign(global_step+1)

mnist = read_data_sets('tmp/MNIST_data')

with tf.Session() as sess:
    c, _ = model.encoder(x)
    x_, _ = model.decoder(c)
    loss = model.loss(x, x_)
    train_op = model.train(loss)

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    if restore_model_path:
        saver.restore(sess, restore_model_path)
        print('model restored from "%s"' % (restore_model_path))

    while True:
        eval_step = sess.run(global_step)
        if eval_step % 4096 == 0 and saveto_model_path:
            saver.save(sess, saveto_model_path)
            print('model saved to "%s"' % (saveto_model_path))

        batch = mnist.train.next_batch(50)[0]
        _, eval_loss = sess.run([train_op, loss], feed_dict={x: batch})
        if eval_step % 256 == 0:
            print('step: %d, loss: %g' % (eval_step, eval_loss))

        inc_global_step.op.run()
