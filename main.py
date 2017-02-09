#! -*- coding:utf-8 -*-

import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from Model import Model
from util import get_figs, dump_figs
import subprocess


if __name__ == u'__main__':

    # figs dir
    input_dir = u'train_split/inputs'
    target_dir = u'train_split/targets'

    val_input_dir = u'val_split/inputs'
    val_target_dir = u'val_split/targets'

    # parameter
    batch_size = 10
    epoch_num = 10
    z_dim = 100
    gen_num_in_one_step = 4
    
    # make model
    print('-- make model --')
    model = Model(z_dim, batch_size)
    model.set_model()

    # get_data
    print('-- get figs--')
    input_figs, target_figs = get_figs(input_dir, target_dir)
    assert(len(input_figs) == len(target_figs))
    print('num figs = {}'.format(len(input_figs)))
    
    val_figs, val_targets = get_figs(val_input_dir, val_target_dir, False, False)

    nrr = np.random.RandomState()
    def shuffle(x, y):
        assert x.shape[0] == y.shape[0]
        rand_ix = nrr.permutation(x.shape[0])
        return x[rand_ix], y[rand_ix]    

    # training
    print('-- begin training --')
    num_one_epoch = len(input_figs) //batch_size

    with tf.Session() as sess:
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess.run(init)

        for epoch in range(epoch_num):
            input_figs, target_figs = shuffle(input_figs, target_figs)
            val_figs, val_targets = shuffle(val_figs, val_targets)
            print('** epoch {} begin **'.format(epoch))
            g_obj = 0.0
            d_obj = 0.0
            for step in range(num_one_epoch):
                
                # get batch data
                batch_z = np.random.normal(0.0, 1.0, [batch_size, z_dim]).astype(np.float32)
                batch_in_figs = input_figs[step * batch_size: (step + 1) * batch_size]
                batch_t_figs  = target_figs[step * batch_size: (step + 1) * batch_size]
                # train
                d_obj += model.training_disc(sess, batch_in_figs,
                                             batch_t_figs, batch_z)
                g_obj += model.training_gen(sess, batch_in_figs,
                                            batch_t_figs, batch_z)
                
                for _ in range(gen_num_in_one_step - 1):
                    batch_z = np.random.normal(0.0, 1.0, [batch_size, z_dim]).astype(np.float32)
                    model.training_gen(sess, batch_in_figs,
                                       batch_t_figs, batch_z)
                
                if step%10 == 0:
                    print('   step {}/{} end'.format(step, num_one_epoch));sys.stdout.flush()
                    tmp_figs = model.gen_fig(sess, batch_in_figs, batch_z)
                    dump_figs(np.asarray(tmp_figs), 'sample_result')
                    
                    tmp_figs = model.gen_fig(sess, val_figs[0:batch_size], batch_z)
                    dump_figs(np.asarray(tmp_figs), 'sample_result2')
                    
            print('epoch:{}, d_obj = {}, g_obj = {}'.format(epoch,
                                                            d_obj/num_one_epoch,
                                                            g_obj/num_one_epoch))
            saver.save(sess, './model.dump')
