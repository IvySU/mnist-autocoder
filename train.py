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
argparser.add_argument('--classifier',
                       action='store_true',
                       help='train classifier')
argparser.add_argument('--decoder',
                       action='store_true',
                       help='train decoder')
argparser.add_argument('--both',
                       action='store_true',
                       help='train classifier and decoder')

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
y = tf.placeholder(tf.float32, [None, 10])

global_step = tf.Variable(0, name='global_step', trainable=False)
inc_global_step = global_step.assign(global_step+1)

mnist = read_data_sets('tmp/MNIST_data', one_hot=True)

with tf.Session() as sess:
    f, var_classifier = model.classifier(x, 10)
    loss_classifier = model.loss_classifier(f, y)
    train_classifier_op = model.train(loss_classifier, var_classifier)
    accuracy_classifier = model.eval_classifier(f, y)

    x_, var_decoder = model.decoder(f)
    loss_decoder = model.loss_decoder(x, x_)
    train_decoder_op = model.train(loss_decoder, var_decoder)

    train_both_op = model.train(loss_classifier + loss_decoder)

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

        batch = mnist.train.next_batch(50)
        if args.classifier:
            _, eval_accuracy, eval_loss = sess.run([train_classifier_op, accuracy_classifier, loss_classifier], feed_dict={x: batch[0], y: batch[1]})
            if eval_step % 256 == 0:
                print('<classifier> step: %d, accuracy: %.2f, loss: %g' % (eval_step, eval_accuracy, eval_loss))
        if args.decoder:
            _, eval_loss = sess.run([train_decoder_op, loss_decoder], feed_dict={x: batch[0]})
            if eval_step % 256 == 0:
                print('<decoder> step: %d, loss: %g' % (eval_step, eval_loss))
        if args.both:
            _, eval_accuracy, eval_classifier_loss, eval_decoder_loss = sess.run([train_both_op, accuracy_classifier, loss_classifier, loss_decoder], feed_dict={x: batch[0], y: batch[1]})
            if eval_step % 256 == 0:
                print('<both> step: %d, accuracy: %.2f, loss: %g + %g' % (eval_step, eval_accuracy, eval_classifier_loss, eval_decoder_loss))

        inc_global_step.op.run()
