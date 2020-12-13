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
from tensorflow.keras import datasets, layers, models
import tensorflow as tf
from tensorflow.python.keras import backend as K


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

        """
        self.pred = self.srcnn915
        self.loss = tf.keras.reduce_mean(tf.square(self.labels - self.pred))
        """

        # モデル構築
        # conv1: [9 x 9]のフィルタ、特徴マップ64
        # conv2: [1 x 1]のフィルタ、特徴マップ32
        # conv3: [5 x 5]のフィルタ、特徴マップ1 <-これが高解像度画像
        self.conv1 = kl.Conv2D(64, (9, 9), padding='valid', activation='relu',
                               input_shape=(None, self.image_size, self.image_size, self.c_dim))
        self.conv2 = kl.Conv2D(32, (1, 1), padding='valid', activation='relu')
        self.conv3 = kl.Conv2D(1, (5, 5), padding='valid', activation='relu')

        """
        self.pred = self.call()
        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
        """

    # 順伝搬
    def call(self, inputs):
        h1 = self.conv1(inputs)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)

        return h3

    def train_step(self, config):
        if config.is_train:
            input_setup(config)
        else:
            nx, ny = input_setup(config)

        if config.is_train:
            data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "train.h5")
        else:
            data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "test.h5")

        train_data, train_label = read_data(data_dir)

        if config.is_train:
            print("Training...")

        # 学習
        self.compile(optimizer=tf.keras.optimizers.Adam(),
                     loss=tf.keras.losses.MeanSquaredError(),
                     metrics=[self.psnr])
        his = self.fit(train_data, train_label, batch_size=self.batch_size, epochs=config.epoch)

        print("Saving parameters...")
        model_name = "SRCNN.model"
        checkpoint_dir = os.path.join(self.checkpoint_dir, model_name)
        self.save_weights(checkpoint_dir)
        print("Successfully completed\n\n")

        return his, self

    # PSNR(ピーク信号対雑音比)
    def psnr(self, h3, labels):

        return -10 * K.log(K.mean(K.flatten((h3 - labels)) ** 2)) / np.log(10)


# PSNR, 損失値グラフ出力
def graph_output(history):

    # PSNRグラフ
    plt.plot(history.history['psnr'])
    plt.title('Model PSNR')
    plt.ylabel('PSNR')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()

    # 損失値グラフ
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()