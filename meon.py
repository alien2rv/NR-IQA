from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import numpy as np
from scipy import misc
from scipy import stats
import tensorflow as tf
from tensorflow.contrib.layers.python.layers.layers import gdn

class MEON_eval(object):
    def __init__(self, height=256, width=256, channel=3,
                 dist_num=5, checkpoint_dir='./weights/'):
        """

                Args:
                    height: height of image
                    width: width of image
                    channel: number of color channel
                    dist_num: number of distortion types
                    checkpoint_dir: parameter saving directory

                """
        self.sess = tf.Session()
        self.height = height
        self.width = width
        self.channel = channel
        self.checkpoint_dir = checkpoint_dir
        self.dist_num = dist_num
        self.x = tf.placeholder(tf.float32,
                                shape=[None, self.height, self.width, self.channel],
                                name='test_image')

        self.build_model()

    def build_model(self):

        # params for convolutional layers
        width1 = 5
        height1 = 5
        stride1 = 2
        depth1 = 8

        width2 = 5
        height2 = 5
        stride2 = 2
        depth2 = 16

        width3 = 5
        height3 = 5
        stride3 = 2
        depth3 = 32

        width4 = 3
        height4 = 3
        stride4 = 1
        depth4 = 64

        # params for fully-connected layers
        sub1_fc1 = 128
        sub1_fc2 = self.dist_num

        sub2_fc1 = 256
        sub2_fc2 = self.dist_num

        # convolution layer 1
        with tf.variable_scope('conv1'):
            weights = tf.get_variable(name='weights',
                                      shape=[height1, width1, self.channel, depth1],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable(name='biases',
                                     shape=[depth1],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(1e-4))
            padded_x = tf.pad(self.x, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="Constant", name="padding")
            conv_x = tf.nn.conv2d(input=padded_x, filter=weights, padding='VALID', strides=[1, stride1, stride1, 1],
                                  name='conv_x') + biases
            gdn_x = gdn(inputs=conv_x, inverse=False, data_format='channels_last', name='gdn_x')
            pool_x = tf.nn.max_pool(value=gdn_x, ksize=[1,2,2,1],strides=[1,2,2,1], padding='VALID', name='pool_x')

        # convolution layer 2
        with tf.variable_scope('conv2'):
            weights = tf.get_variable(name='weights',
                                      shape=[height2, width2, depth1, depth2],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable(name='biases',
                                     shape=[depth2],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(1e-4))
            padded_x = tf.pad(pool_x, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="Constant", name="padding")
            conv_x = tf.nn.conv2d(input=padded_x, filter=weights, padding='VALID', strides=[1, stride2, stride2, 1],
                                  name='conv_x') + biases
            gdn_x = gdn(inputs=conv_x, inverse=False, data_format='channels_last', name='gdn_x')
            pool_x = tf.nn.max_pool(value=gdn_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool_x')

        # convolution layer 3
        with tf.variable_scope('conv3'):
            weights = tf.get_variable(name='weights',
                                      shape=[height3, width3, depth2, depth3],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable(name='biases',
                                     shape=[depth3],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(1e-4))
            padded_x = tf.pad(pool_x, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="Constant", name="padding")
            conv_x = tf.nn.conv2d(input=padded_x, filter=weights, padding='VALID', strides=[1, stride3, stride3, 1],
                                  name='conv_x') + biases
            gdn_x = gdn(inputs=conv_x, inverse=False, data_format='channels_last', name='gdn_x')
            pool_x = tf.nn.max_pool(value=gdn_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool_x')

        # convolution layer 4
        with tf.variable_scope('conv4'):
            weights = tf.get_variable(name='weights',
                                      shape=[height4, width4, depth3, depth4],
                                      dtype=tf.float32,
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable(name='biases',
                                     shape=[depth4],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(1e-4))

            conv_x = tf.nn.conv2d(input=pool_x, filter=weights, padding='VALID', strides=[1, stride4, stride4, 1],
                                  name='conv_x') +biases
            gdn_x = gdn(inputs=conv_x, inverse=False, data_format='channels_last', name='gdn_x')
            conv_out_x = tf.nn.max_pool(value=gdn_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool_x')

        # subtask 1
        with tf.variable_scope('subtask1'):
            with tf.variable_scope('fc1'):
                weights = tf.get_variable(name='weights',
                                          shape=[1,1,depth4, sub1_fc1],
                                          dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(depth4))))
                biases = tf.get_variable(name='biases',
                                         shape=[sub1_fc1],
                                         dtype=tf.float32,
                                         initializer=tf.constant_initializer(1e-4))
                fc_x = tf.nn.conv2d(input=conv_out_x, filter=weights, padding='VALID', strides=[1,1,1,1], name='fc_x') +biases
                gdn_x = gdn(inputs=fc_x, inverse=False, data_format='channels_last', name='gdn_x')

            with tf.variable_scope('fc2'):
                weights = tf.get_variable(name='weights',
                                          shape=[1,1, sub1_fc1,sub1_fc2],
                                          dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(sub1_fc1))))
                biases = tf.get_variable(name='biases',
                                         shape=[sub1_fc2],
                                         dtype=tf.float32,
                                         initializer=tf.constant_initializer(1e-4))
                out_x = tf.squeeze(tf.nn.conv2d(input=gdn_x, filter=weights, padding='VALID', strides=[1,1,1,1])+biases,
                                   name='out_x')

            self.probs = tf.nn.softmax(out_x,name='dist_prob')

        # subtask 2
        with tf.variable_scope('subtask2'):
            with tf.variable_scope('fc1'):
                weights = tf.get_variable(name='weights',
                                          shape=[1,1,depth4, sub2_fc1],
                                          dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(depth4))))
                biases = tf.get_variable(name='biases',
                                         shape=[sub2_fc1],
                                         dtype=tf.float32,
                                         initializer=tf.constant_initializer(1e-4))
                fc_x = tf.nn.conv2d(input=conv_out_x, filter=weights, padding='VALID', strides=[1, 1, 1, 1],
                                    name='fc_x') + biases
                gdn_x = gdn(inputs=fc_x , inverse=False, data_format='channels_last', name='gdn_x')

            with tf.variable_scope('fc2'):
                weights = tf.get_variable(name='weights',
                                          shape=[1,1,sub2_fc1, sub2_fc2],
                                          dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(
                                              stddev=1.0 / math.sqrt(float(sub2_fc1))))
                biases = tf.get_variable(name='biases',
                                         shape=[sub2_fc2],
                                         dtype=tf.float32,
                                         initializer=tf.constant_initializer(1e-4))
                self.q_scores = tf.squeeze(tf.nn.conv2d(input=gdn_x, filter=weights, padding='VALID',
                                                        strides=[1,1,1,1])+biases, name='q_scores')
                self.out_q = tf.reduce_sum(tf.multiply(self.probs, self.q_scores),
                                           axis=1, keep_dims=False, name='out_q')

            self.saver = tf.train.Saver()

MEON_evaluate_model = MEON_eval()

with tf.Session() as sess:
    writer = tf.summary.FileWriter("output", sess.graph)
    print(sess.run(q_scores))
    writer.close()
