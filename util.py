#! -*- coding:utf-8 -*-

import os
import sys

import tensorflow as tf
import cv2
import numpy as np

def get_weights(name, shape, stddev, trainable = True):
    return tf.get_variable('weights{}'.format(name), shape,
                           initializer = tf.random_normal_initializer(stddev = stddev),
                           trainable = trainable)

def get_biases(name, shape, value, trainable = True):
    return tf.get_variable('biases{}'.format(name), shape,
                           initializer = tf.constant_initializer(value),
                           trainable = trainable)

def get_dim(target):
    dim = 1
    for d in target.get_shape()[1:].as_list():
        dim *= d
    return dim



def random_jitter(img, offset):
    length = len(img)
    
    re_img = cv2.resize(img, (length + offset, length + offset))
    target = np.random.choice(range(length + offset), offset, replace = False)
    tmp = np.delete(re_img, target, 1)
    target = np.random.choice(range(length + offset), offset, replace = False)
    ret = np.delete(tmp, target, 0)
    return ret

def get_figs(input_dir, target_dir, mirroring = True, jitter = True):
    input_ret = []
    target_ret = []
    
    for file_name in os.listdir(input_dir):
        tmp = cv2.imread(os.path.join(input_dir, file_name))
        tmp_ = cv2.imread(os.path.join(target_dir, file_name))
        if tmp_ is None:
            continue
        tmp = tmp/127.5 - 1.0
        tmp_ = tmp_/127.5 - 1.0
        input_ret.append(tmp)
        target_ret.append(tmp_)
        
        if mirroring:
            input_ret.append(cv2.flip(tmp, 0))
            target_ret.append(cv2.flip(tmp_, 0))
            
            input_ret.append(cv2.flip(tmp, 1))
            target_ret.append(cv2.flip(tmp_, 1))
            
        if jitter:
            for i in range(3):
                input_ret.append(random_jitter(tmp, (i + 1) * 30))
                target_ret.append(tmp_)
        
    return np.asarray(input_ret, dtype = np.float32), np.asarray(target_ret, dtype = np.float32)

def dump_figs(imgs, dir_name):
    for i, img in enumerate(imgs):
        cv2.imwrite(os.path.join(dir_name, '{}.jpg'.format(i)), (img + 1.0) * 127.5)


        
def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def linear_layer(inputs, in_dim, out_dim, name):
    w = get_weights(name, [in_dim, out_dim], 0.02)
    b = get_biases(name, [out_dim], 0.0)
    return tf.matmul(inputs, w) + b

def conv_layer(inputs, out_num, filter_width, filter_hight, stride, l_id):
    # ** NOTICE: weight shape is [hight, width, in_chanel, out_chanel] **
    weights = get_weights(l_id,
                          [filter_hight, filter_width, inputs.get_shape()[-1], out_num],
                          0.02)
    
    biases = get_biases(l_id, [out_num], 0.0)
    
    conved = tf.nn.conv2d(inputs, weights,
                          strides=[1, stride,  stride,  1],
                          padding = 'SAME')
    
    return tf.nn.bias_add(conved, biases)


def deconv_layer(inputs, out_shape, filter_width, filter_hight, stride, l_id):
    # ** NOTICE: weight shape is [hight, width, out_chanel, in_chanel] **
    weights = get_weights(l_id,
                          [filter_hight, filter_width, out_shape[-1], inputs.get_shape()[-1]],
                          0.02)
    
    biases = get_biases(l_id, [out_shape[-1]], 0.0)
    
    deconved = tf.nn.conv2d_transpose(inputs, weights, output_shape = out_shape,
                                      strides=[1, stride,  stride,  1])
    return tf.nn.bias_add(deconved, biases)
