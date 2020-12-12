from utils import (
  read_data,
  input_setup,
  imsave,
  merge
)

import os
import glob
import h5py
import random
import matplotlib.pyplot as plt


import scipy.misc
import scipy.ndimage
import numpy as np

import tensorflow.keras.layers as kl
import tensorflow as tf


# SRCNN
class SRCNN(tf.keras.Model):

    def __init__(self,
                 image_size=33,
                 label_size=21,
                 batch_size=128,
                 c_dim=1,
                 checkpoint_dir=None,
                 sample_dir=None):

        super(SRCNN, self).__init__()

        self.is_grayscale = (c_dim == 1)
        self.image_size = image_size
        self.label_size = label_size
        self.batch_size = batch_size

        self.c_dim = c_dim

        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir

        # モデル構築
        # conv1: 9*9のフィルタ、特徴マップ64
        # conv2: 1*1のフィルタ、特徴マップ32
        # conv3: 5*5のフィルタ、特徴マップ1 <-これが高解像度画像
        self.conv1 = kl.Conv2D(64, (9, 9), padding='same', activation='relu',
                               input_shape=(None, self.image_size, self.image_size, self.c_dim))
        self.conv2 = kl.Conv2D(32, (1, 1), padding='same', activation='relu')
        self.conv3 = kl.Conv2D(1, (5, 5), padding='same', activation='relu')

        self.pred = self.model()
        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))

        self.compile(optimizer=tf.keras.optimizers.Adam(),
                     loss=tf.keras.losses.MeanSquaredError(),
                     metrics=[self.loss])

    # 順伝搬
    def model(self):
        h1 = self.conv1(self.images)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)

        return h3

    def train(self, config):
        if config.is_train:
            input_setup(config)
        else:
            nx, ny = input_setup(self.sess, config)

        if config.is_train:
            data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "train.h5")
        else:
            data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "test.h5")

        train_data, train_label = read_data(data_dir)