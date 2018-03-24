"""
Train MNIST triplet loss
"""

import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils.data_serve import triplet_load_train, triplet_load_validation, test_images
from network import Network

model_path = "saved_models/model.ckpt"
embedding_path = "saved_models/embedding.bin"

# Prepare Session
sess = tf.InteractiveSession()

# setup siamese network
network = Network();
train_op = tf.train.AdamOptimizer(0.001).minimize(network.loss)
# saver = tf.train.Saver()
tf.global_variables_initializer().run()

saver = tf.train.Saver()

try:
    saver.restore(sess, model_path)
    print("Model restored from file: {}".format(model_path))
except:
    print("Could not restore saved model")

step = 0
try:
    while True:
        anchor, positive, negative = triplet_load_train(128)

        _, loss_value = sess.run([train_op, network.loss], feed_dict={
                            network.anchor: anchor,
                            network.positive: positive,
                            network.negative: negative,
                            network.is_training: True})

        if np.isnan(loss_value):
            raise ValueError('Loss value is NaN')

        if step % 10 == 0:
            anchor, positive, negative = triplet_load_validation(1024)
            loss_validation = sess.run(network.loss, feed_dict={
                            network.anchor: anchor,
                            network.positive: positive,
                            network.negative: negative,
                            network.is_training: False})

            print ('step %d: training loss %.3f validation loss %.3f' % (step, loss_value, loss_validation))

        if step % 100 == 0:
            save_path = saver.save(sess, model_path)
            embed = network.i_vector3.eval({network.negative: test_images(), network.is_training: False})
            embed.tofile(embedding_path)

        step+=1

except (KeyboardInterrupt, SystemExit):
    print("Manual Interrupt")
    save_path = saver.save(sess, model_path)
    embed = network.i_vector3.eval({network.negative: test_images(), network.is_training: False})
    embed.tofile(embedding_path)

except Exception as e:
    print("Exception: {}".format(e))
