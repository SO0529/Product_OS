from utils import (
    read_data,
    input_setup,
    compare_res_and_label,
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
                 checkpoint_dir=None):

        super(SRCNN, self).__init__()

        self.is_grayscale = (c_dim == 1)
        self.image_size = image_size
        self.label_size = label_size
        self.batch_size = batch_size

        self.c_dim = c_dim

        self.checkpoint_dir = checkpoint_dir

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
            print("invalid config")
            return

        if config.is_train:
            data_dir = os.path.join('./{}'.format(config.h5_dir), "train.h5")
        else:
            print("invalid config")
            return

        """
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        """
        train_data, train_label = read_data(data_dir)

        print("Training...")

        # チェックポイント（100エポックずつ保存）
        checkpoint_path = '%s/cp-x%s-{epoch:04d}.ckpt' % (self.checkpoint_dir, config.scale)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1,
                                                         period=100)

        # 学習
        self.compile(optimizer=tf.keras.optimizers.Adam(),
                     loss=tf.keras.losses.MeanSquaredError(),
                     metrics=[self.psnr])

        self.save_weights(checkpoint_path.format(epoch=0))

        his = self.fit(train_data,
                       train_label,
                       batch_size=self.batch_size,
                       epochs=config.epoch,
                       callbacks=[cp_callback],
                       verbose=1)

        print("Successfully completed\n\n")

        return his, self

    # Test for eval (指定したフォルダ内の画像すべてに超解像（現在はとりあえず1枚だけ）)
    def test_all(self, config):

        if config.is_train:
            print("invalid config")
            return
        else:
            nx, ny = input_setup(config)

        if config.is_train:
            print("invalid config")
            return
        else:
            data_dir = os.path.join('./{}'.format(config.h5_dir), "test.h5")

        test_data, test_label = read_data(data_dir)

        latest = tf.train.latest_checkpoint(self.checkpoint_dir)
        self.load_weights(latest)

        print("Testing...")

        # 高解像度画像作成
        result = self.predict(test_data)
        result = merge(result, [nx, ny])
        result = result.squeeze()
        result *= 255

        # 正解画像復元
        label = merge(test_label, [nx, ny])
        label = label.squeeze()
        label *= 255

        # 結果と正解をGrayscaleで表示
        compare_res_and_label(result, label, True)
        res_savepath = os.path.join('./{}'.format(config.save_dir), "result.bmp")
        gt_savepath = os.path.join('./{}'.format(config.save_dir), "gt.bmp")
        imsave(res_savepath, result)
        imsave(gt_savepath, label)

    # PSNR(ピーク信号対雑音比)
    def psnr(self, h3, labels):
        return -10 * K.log(K.mean(K.flatten((h3 - labels)) ** 2)) / np.log(10)
