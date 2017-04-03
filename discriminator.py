#! -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf

from util import conv_layer, get_dim, lrelu, linear_layer
from batch_normalize import batch_norm


class Discriminator(object):
    def __init__(self, out_dims):
        self.out_dims = out_dims
        
        self.name_scope_conv = u'convolution'
        self.name_scope_fc = u'full_connect'
        
    def get_variables(self):
        t_var = tf.trainable_variables()
        ret = []
        for var in t_var:
            if self.name_scope_conv in var.name or self.name_scope_fc in var.name:
                ret.append(var)
        return ret
    
    def set_model(self, figs, is_training, reuse = False):

        u'''
        return only logits. not sigmoid(logits).
        '''
        
        h = figs
        
        # convolution
        with tf.variable_scope(self.name_scope_conv, reuse = reuse):
            for i, out_dim in enumerate(self.out_dims):
                conved = conv_layer(inputs = h,
                                    out_num = out_dim,
                                    filter_width = 4, filter_hight = 4,
                                    stride = 2, l_id = i)
                if i == 0:
                    #h = tf.nn.relu(conved)
                    h = lrelu(conved)
                else:
                    bn_conved = batch_norm(conved, i, is_training)
                    #h = tf.nn.relu(bn_conved)
                    h = lrelu(bn_conved)

        # full connect
        dim = get_dim(h)
        h = tf.reshape(h, [-1, dim])
        with tf.variable_scope(self.name_scope_fc, reuse = reuse):
            h = linear_layer(h, dim, 1, 'fc')
            
        return h
    
if __name__ == u'__main__':
    g = Discriminator([64, 128, 256, 512])
    figs = tf.placeholder(tf.float32, [None, 256, 256, 3])
    t_figs = tf.placeholder(tf.float32, [None, 256, 256, 3])
    g.set_model(tf.concat([figs, t_figs], 3), True)
