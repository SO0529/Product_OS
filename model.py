from utils import (
    read_data,
    input_setup,
    compare_res_and_label,
    imsave,
    merge
)

import os
import csv
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
            nx, ny, dataname = input_setup(config)

        if config.is_train:
            print("invalid config")
            return
        else:
            data_dir = os.path.join('./{}'.format(config.h5_dir), "test.h5")

        test_data, test_label = read_data(data_dir)

        latest = tf.train.latest_checkpoint(self.checkpoint_dir)
        self.load_weights(latest)

        print("Testing...")

        # 高解像度画像作成（全データ結合状態）
        result = self.predict(test_data)

        # 高解像度画像と正解画像を分割してそれぞれ1枚ずつの画像にする(関数化は後回し)
        split_result = []
        split_bc = []
        split_label = []
        psnr_list = []
        psnr_bc_list = []
        n_start = 0     # 統合状態での各画像のスタートポイント
        for i in range(len(nx)):
            if i == 0:
                t_res = result[:nx[i]*ny[i], :, :, :]
                shave = (test_data.shape[2] - t_res.shape[2]) // 2
                t_bc = test_data[:nx[i] * ny[i], shave:-shave, shave:-shave, :]
                t_label = test_label[:nx[i]*ny[i], :, :, :]
                n_start = nx[i]*ny[i]
            else:
                t_res = result[n_start:n_start + nx[i] * ny[i], :, :, :]
                shave = (test_data.shape[2] - t_res.shape[2]) // 2
                t_bc = test_data[n_start:n_start + nx[i] * ny[i], shave:-shave, shave:-shave, :]
                t_label = test_label[n_start:n_start + nx[i] * ny[i], :, :, :]
                n_start += nx[i] * ny[i]
            split_result.append(t_res)
            split_bc.append(t_bc)
            split_label.append(t_label)
            split_result[i] = merge(split_result[i], [nx[i], ny[i]])
            split_bc[i] = merge(split_bc[i], [nx[i], ny[i]])
            split_label[i] = merge(split_label[i], [nx[i], ny[i]])
            split_result[i] = split_result[i].squeeze()
            split_bc[i] = split_bc[i].squeeze()
            split_label[i] = split_label[i].squeeze()

            # PSNR計算
            psnr_list.append(K.get_value(self.psnr(split_result[i], split_label[i])))
            psnr_bc_list.append(K.get_value(self.psnr(split_bc[i], split_label[i])))

            split_result[i] *= 255
            split_bc[i] *= 255
            split_label[i] *= 255

            # 結果と正解をGrayscaleで表示
            compare_res_and_label(split_result[i], split_label[i], True)
            res_savepath = './' + config.save_dir + '/' + os.path.splitext(os.path.basename(dataname[i]))[0] + \
                           '_res_x%s.bmp' % config.scale
            bc_savepath = './' + config.save_dir + '/' + os.path.splitext(os.path.basename(dataname[i]))[0] + \
                           '_bc_x%s.bmp' % config.scale
            gt_savepath = './' + config.save_dir + '/' + os.path.splitext(os.path.basename(dataname[i]))[0] + \
                          '_gt_x%s.bmp' % config.scale
            imsave(res_savepath, split_result[i])
            imsave(bc_savepath, split_bc[i])
            imsave(gt_savepath, split_label[i])

        with open(config.save_dir + '/psnr.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(psnr_list)
            writer.writerow(psnr_bc_list)

    # PSNR(ピーク信号対雑音比)
    def psnr(self, h3, labels):
        return -10 * K.log(K.mean(K.flatten((h3 - labels)) ** 2)) / np.log(10)
