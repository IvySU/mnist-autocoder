#!/usr/bin/env python3

import os
import argparse
import random

import tensorflow as tf
import numpy as np

import model


def mat2str(name, M):
    lines = []
    for v in M:
        lines.append('[' + ', '.join([str(x) for x in v]) + ']')
    return 'var ' + name + ' = \n' + '[' + ',\n'.join(lines) + ']' + ';\n'


def vec2str(name, V):
    return 'var ' + name + ' = \n[' + ', '.join([str(x) for x in V]) + '];\n'


argparser = argparse.ArgumentParser(description='MNIST Autocoder')
argparser.add_argument('--model', metavar='<model name>',
                       type=str, default='coder',
                       help='model to test')
argparser.add_argument('--file', metavar='<file name>',
                       type=str,
                       help='file name')

args = argparser.parse_args()

model_path = os.path.join('model', args.model + '.ckpt')

if args.file:
    file_path = args.file
else:
    file_path = args.model + '.js'


x = tf.placeholder(tf.float32, [None, 28*28])

with tf.Session() as sess:
    c, encoder_vars = model.encoder(x)
    _, decoder_vars = model.decoder(c)

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    print('"%s" loaded' % (model_path))

    eval_encoder_vars, eval_decoder_vars = sess.run([encoder_vars, decoder_vars])

    with open(file_path, "w") as handle:
        handle.write(mat2str('encoder_W', eval_encoder_vars[0]))
        handle.write(vec2str('encoder_b', eval_encoder_vars[1]))
        handle.write(mat2str('decoder_W', eval_decoder_vars[0]))
        handle.write(vec2str('decoder_b', eval_decoder_vars[1]))
