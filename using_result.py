#! -*- coding:utf-8 -*-

import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

from Model import Model
from util import get_figs, dump_figs

class FigGenerator(object):
    def __init__(self, file_name, z_dim, batch_size):

        self.batch_size = batch_size
        self.z_dim = z_dim
        self.model = Model(z_dim, batch_size)
        self.model.set_model()
        saver = tf.train.Saver()
        self.sess = tf.Session()

        saver.restore(self.sess, file_name)
        
    def __call__(self, inputs):
        assert(len(inputs) == self.batch_size)
        #z = np.zeros([self.batch_size, self.z_dim])
        z =  np.random.normal(0.0, 1.0, [batch_size, z_dim]).astype(np.float32)
        return self.model.gen_fig(self.sess, inputs, z)

if __name__ == u'__main__':

    # dump file
    dump_file = u'./model.dump'

    # dir
    input_dir = u'train_split/inputs'
    target_dir = u'train_split/targets'
    # parameter
    batch_size = 10
    z_dim = 100

    # figure generator
    fig_gen = FigGenerator(dump_file, z_dim, batch_size)

    # get fig
    print('-- get figs--')
    input_figs, target_figs = get_figs(input_dir, target_dir, False, False)
    assert(len(input_figs) == len(target_figs))
    print('num figs = {}'.format(len(input_figs)))

    # make figure
    inputs =  input_figs[10: 10 + batch_size]
    input_imgs = cv2.hconcat((inputs + 1.0) * 127.5)
    cv2.imwrite('inputs.jpg', input_imgs)
    
    outputs = np.asarray(fig_gen(input_figs[10: 10 +batch_size]))
    output_imgs = cv2.hconcat((outputs + 1.0) * 127.5)
    cv2.imwrite('outputs.jpg',  output_imgs)
