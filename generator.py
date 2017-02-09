#! -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf

from util import deconv_layer, conv_layer, lrelu, linear_layer
from batch_normalize import batch_norm

class Generator(object):
    def __init__(self, encode_out_dims, decode_out_dims, in_width, in_hight, in_chan):
        self.encode_out_dims = encode_out_dims
        self.decode_out_dims = decode_out_dims
        self.in_width = in_width
        self.in_hight = in_hight
        self.in_chan = in_chan
        
        self.name_scope_reshape = u'reshape'
        self.name_scope_encode  = u'encode'
        self.name_scope_decode  = u'decode'

    def get_variables(self):
        t_var = tf.trainable_variables()
        ret = []
        for var in t_var:
            if self.name_scope_decode in var.name or self.name_scope_encode in var.name or self.name_scope_reshape in var.name:
                ret.append(var)
        return ret
    
    def set_model(self, input_figs, z, batch_size, is_training):
        u'''
        # reshape z
        with tf.variable_scope(self.name_scope_reshape):
            z_dim = z.get_shape().as_list()[1]
            h = linear_layer(z, z_dim, self.in_width * self.in_hight * self.in_chan, 'reshape')
            h = batch_norm(h, 'reshape', is_training)
            h = tf.nn.relu(h)
            #h = lrelu(h)
            
        h = tf.reshape(h, [-1, self.in_width, self.in_hight, self.in_chan])

        h = tf.concat(3, [input_figs, h])
        '''
        h = input_figs
        encoded_list = []
        
        # encode
        with tf.variable_scope(self.name_scope_encode):
            for i, out_dim in enumerate(self.encode_out_dims):
                h = conv_layer(h, out_dim, 4, 4, 2, i)
                if i == 0:
                    encoded_list.append(h)
                    h = lrelu(h)
                else:
                    h = batch_norm(h, i, is_training)
                    encoded_list.append(h)
                    h = lrelu(h)
            
        # deconvolution
        encoded_list.pop()
        h = tf.nn.relu(h)
        with tf.variable_scope(self.name_scope_decode):
            for i, out_dim in enumerate(self.decode_out_dims):
                hight = 2 * h.get_shape().as_list()[1]
                width = 2 * h.get_shape().as_list()[2]
                h = deconv_layer(inputs = h,
                                 out_shape = [batch_size, width, hight, out_dim],
                                 filter_width = 4, filter_hight = 4,
                                 stride = 2, l_id = i)
                h = batch_norm(h, i, is_training)
                #if is_training and i <= 2:
                if i <= 2:
                    h = tf.nn.dropout(h, 0.5)
                h = tf.concat(3, [h, encoded_list.pop()])
                h = tf.nn.relu(h)

            hight = 2 * h.get_shape().as_list()[1]
            width = 2 * h.get_shape().as_list()[2]                
            h = deconv_layer(inputs = h,
                             out_shape = [batch_size, width, hight, 3],
                             filter_width = 4, filter_hight = 4,
                             stride = 2, l_id = len(self.decode_out_dims))
        return tf.nn.tanh(h)
    
if __name__ == u'__main__':
    g = Generator([64, 128, 256, 512, 512, 512, 512, 512],
                  [512, 512, 512, 512, 256, 128, 64],
                  256, 256, 3)
    figs = tf.placeholder(tf.float32, [None, 256, 256, 3])
    z = tf.placeholder(tf.float32, [None, 256])
    g.set_model(figs, z, 100, True)

