#! -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import tensorflow as tf

from generator import Generator
from discriminator import Discriminator
    
class Model(object):
    def __init__(self,  z_dim, batch_size):
        self.batch_size = batch_size
        self.z_dim = z_dim
        
        # -- generator -----
        self.gen = Generator([64, 128, 256, 512, 512, 512, 512, 512],
                             [512, 512, 512, 512, 256, 128, 64],
                             256, 256, 3)

        # -- discriminator --
        self.disc = Discriminator([64, 128, 256, 512])

        # -- learning parms ---
        self.lr = 0.0002
        self.Lambda = 100.0
        
    def set_model(self):
        self.input_figs  = tf.placeholder(tf.float32, [self.batch_size, 256, 256, 3])
        self.target_figs = tf.placeholder(tf.float32, [self.batch_size, 256, 256, 3])
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim])
        
        # -- input -> gen_fig -> disc ---
        gen_figs = self.gen.set_model(self.input_figs, self.z, self.batch_size,
                                      is_training = True)
        l1_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(gen_figs - self.target_figs), [1, 2, 3]))

        g_logits = self.disc.set_model(tf.concat(3, [gen_figs, self.target_figs]),
                                       is_training = True)
        self.g_obj = -tf.reduce_mean(tf.reduce_sum(tf.log(1.0e-6 + tf.nn.sigmoid(g_logits))))
        self.g_obj += self.Lambda * l1_loss
        
        self.train_gen  = tf.train.AdamOptimizer(self.lr, beta1 = 0.5).minimize(self.g_obj, var_list = self.gen.get_variables())
        
        # -- for sharing variables ---
        tf.get_variable_scope().reuse_variables()
        
        # -- input + target -> disc --------
        d_logits = self.disc.set_model(tf.concat(3, [self.input_figs, self.target_figs]),
                                       is_training = True)
        
        d_obj_true = -tf.reduce_mean(tf.reduce_sum(tf.log(1.0e-6 + tf.nn.sigmoid(d_logits))))
        d_obj_fake = -tf.reduce_mean(tf.reduce_sum(tf.log(1.0e-6 + 1 - tf.nn.sigmoid(g_logits))))
        self.d_obj = d_obj_true + d_obj_fake

        self.train_disc = tf.train.AdamOptimizer(self.lr, beta1 = 0.5).minimize(self.d_obj, var_list = self.disc.get_variables())

        # -- for figure generation -------
        self.gen_figs = self.gen.set_model(self.input_figs,self.z, self.batch_size,
                                           is_training = False)
        
    def training_gen(self, sess, inputs, targets, z_list):
        _, g_obj = sess.run([self.train_gen, self.g_obj],
                            feed_dict = {self.input_figs:inputs,
                                         self.target_figs:targets,
                                         self.z: z_list})
        return g_obj
        
    def training_disc(self, sess, inputs, targets, z_list):
        _, d_obj = sess.run([self.train_disc, self.d_obj],
                            feed_dict = {self.z: z_list,
                                         self.input_figs:inputs,
                                         self.target_figs:targets})
        return d_obj
    
    def gen_fig(self, sess, inputs, z):
        ret = sess.run(self.gen_figs,
                       feed_dict = {self.input_figs:inputs,
                                    self.z: z})
        return ret

if __name__ == u'__main__':
    model = Model(z_dim = 100, batch_size = 100)
    model.set_model()
    
