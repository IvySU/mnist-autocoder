import tensorflow as tf


def encoder(x):
    var_list = []
    output = x

    with tf.name_scope('encoder/fc0'):
        weights = tf.Variable(tf.truncated_normal(shape=[int(output.shape[1]), 256]),
                              name='weights')
        biases = tf.Variable(tf.zeros(shape=[256]),
                             name='biases')
        output = tf.nn.sigmoid(tf.matmul(output, weights) + biases)
        var_list += [weights, biases]

    with tf.name_scope('encoder/fc1'):
        weights = tf.Variable(tf.truncated_normal(shape=[int(output.shape[1]), 128]),
                              name='weights')
        biases = tf.Variable(tf.zeros(shape=[128]),
                             name='biases')
        output = tf.nn.sigmoid(tf.matmul(output, weights) + biases)
        var_list += [weights, biases]

    with tf.name_scope('encoder/output'):
        weights = tf.Variable(tf.truncated_normal(shape=[int(output.shape[1]), 64]),
                              name='weights')
        biases = tf.Variable(tf.zeros(shape=[64]),
                             name='biases')
        output = tf.nn.sigmoid(tf.matmul(output, weights) + biases)
        var_list += [weights, biases]

    return output, var_list


def decoder(c):
    var_list = []
    output = c

    with tf.name_scope('decoder/fc0'):
        weights = tf.Variable(tf.truncated_normal(shape=[int(output.shape[1]), 128]),
                              name='weights')
        biases = tf.Variable(tf.zeros(shape=[128]),
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

    with tf.name_scope('decoder/output'):
        weights = tf.Variable(tf.truncated_normal(shape=[int(output.shape[1]), 28*28]),
                              name='weights')
        biases = tf.Variable(tf.zeros(shape=[28*28]),
                             name='biases')
        output = tf.nn.sigmoid(tf.matmul(output, weights) + biases)
        var_list += [weights, biases]

    return output, var_list


def loss(x, x_):
    return tf.reduce_mean(tf.square(x - x_))


def train(loss, var_list=None):
    optimizer = tf.train.AdamOptimizer(1e-4)
    return optimizer.minimize(loss, var_list=var_list)
