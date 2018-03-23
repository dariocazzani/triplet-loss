import tensorflow as tf
from utils.tf_ops import compute_triplet_loss


class Network(object):

    # Create model
    def __init__(self):
        self.anchor = tf.placeholder(tf.float32, [None, 784])
        self.positive = tf.placeholder(tf.float32, [None, 784])
        self.negative = tf.placeholder(tf.float32, [None, 784])
        self.margin = 0.1
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        with tf.variable_scope("Network") as scope:
            self.i_vector1 = self.network(self.anchor, False)
            self.i_vector2 = self.network(self.positive, True)
            self.i_vector3 = self.network(self.negative, True)

        # Create loss
        self.y_ = tf.placeholder(tf.float32, [None])
        self.loss = self.compute_loss()

    def network(self, x, reuse):
        inputs = tf.reshape(x, [-1, 28, 28, 1])
        inputs = tf.layers.conv2d(inputs, 16, [5, 5], reuse=reuse, activation=tf.nn.relu, name='cnn1')
        inputs = tf.layers.conv2d(inputs, 32, [3, 3], reuse=reuse, activation=tf.nn.relu, name='cnn2')
        inputs = tf.layers.conv2d(inputs, 64, [3, 3], reuse=reuse, activation=tf.nn.relu, name='cnn3')
        inputs = tf.layers.flatten(inputs)
        inputs = tf.layers.dense(inputs, 128, reuse=reuse, activation=tf.nn.relu, name='fc1')
        inputs = tf.layers.dropout(inputs, rate=0.5, training=self.is_training)
        inputs = tf.layers.dense(inputs, 2, reuse=reuse, activation=None, name='fc2')
        return inputs

    def compute_loss(self):
        return compute_triplet_loss(self.i_vector1, self.i_vector2, self.i_vector3, self.margin)
