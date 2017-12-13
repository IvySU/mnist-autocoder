import tensorflow as tf
import numpy as np


def classifier(x, n_class):
    var_list = []
    output = x

    with tf.name_scope('classifier/fc0'):
        weights = tf.Variable(tf.truncated_normal(shape=[int(output.shape[1]), 256]),
                              name='weights')
        biases = tf.Variable(tf.zeros(shape=[256]),
                             name='biases')
        output = tf.nn.sigmoid(tf.matmul(output, weights) + biases)
        var_list += [weights, biases]

    with tf.name_scope('classifier/fc1'):
        weights = tf.Variable(tf.truncated_normal(shape=[int(output.shape[1]), 256]),
                              name='weights')
        biases = tf.Variable(tf.zeros(shape=[256]),
                             name='biases')
        output = tf.nn.sigmoid(tf.matmul(output, weights) + biases)
        var_list += [weights, biases]

    with tf.name_scope('classifier/fc2'):
        weights = tf.Variable(tf.truncated_normal(shape=[int(output.shape[1]), 256]),
                              name='weights')
        biases = tf.Variable(tf.zeros(shape=[256]),
                             name='biases')
        output = tf.nn.sigmoid(tf.matmul(output, weights) + biases)
        var_list += [weights, biases]

    with tf.name_scope('classifier/output'):
        weights = tf.Variable(tf.truncated_normal(shape=[int(output.shape[1]), 10]),
                              name='weights')
        biases = tf.Variable(tf.zeros(shape=[10]),
                             name='biases')
        output = tf.matmul(output, weights) + biases
        var_list += [weights, biases]

    return output, var_list


def loss_classifier(scores, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=scores,
                                                            name='cross_entropy')
    return tf.reduce_mean(cross_entropy, name='cross_entropy_mean')


def decoder(feature):
    var_list = []
    output = tf.nn.sigmoid(feature)

    with tf.name_scope('decoder/fc0'):
        weights = tf.Variable(tf.truncated_normal(shape=[int(output.shape[1]), 256]),
                              name='weights')
        biases = tf.Variable(tf.zeros(shape=[256]),
                             name='biases')
        output = tf.nn.sigmoid(tf.matmul(output, weights) + biases)
        var_list += [weights, biases]

    with tf.name_scope('decoder/fc1'):
        weights = tf.Variable(tf.truncated_normal(shape=[int(output.shape[1]), 256]),
                              name='weights')
        biases = tf.Variable(tf.zeros(shape=[256]),
                             name='biases')
        output = tf.nn.sigmoid(tf.matmul(output, weights) + biases)
        var_list += [weights, biases]

    with tf.name_scope('decoder/fc2'):
        weights = tf.Variable(tf.truncated_normal(shape=[int(output.shape[1]), 256]),
                              name='weights')
        biases = tf.Variable(tf.zeros(shape=[256]),
                             name='biases')
        output = tf.nn.sigmoid(tf.matmul(output, weights) + biases)
        var_list += [weights, biases]

    with tf.name_scope('decoder/output'):
        weights = tf.Variable(tf.truncated_normal(shape=[int(output.shape[1]), 28*28]),
                              name='weights')
        biases = tf.Variable(tf.zeros(shape=[28*28]),
                             name='biases')
        output = tf.nn.sigmoid(tf.matmul(output, weights) + biases)
        var_list += [weights, biases]

    return output, var_list


def loss_decoder(x, x_):
    return tf.reduce_mean(tf.square(x - x_))


def train(loss, var_list=None):
    optimizer = tf.train.AdamOptimizer(1e-4)
    return optimizer.minimize(loss, var_list=var_list)


def eval_classifier(scores, labels):
    correct = tf.equal(tf.argmax(scores, 1), tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(correct, tf.float32))
