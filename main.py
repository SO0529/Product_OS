from model import SRCNN
from utils import input_setup

import numpy as np
from absl import app
from absl import flags
from tensorflow.keras import layers
from tensorflow.keras import datasets, layers, models
import tensorflow as tf

import matplotlib.pyplot as plt
import pprint
import os
import datetime

# 33*33の入力が9*1*5の畳み込みで画像周り6pix分小さくなるため、ラベルは21*21となる
# 学習時は"is_train"をFalseにする
FLAGS = flags.FLAGS
flags.DEFINE_integer("epoch", 1000, "Number of epoch []")
flags.DEFINE_integer("batch_size", 128, "The size of batch images [128]")
flags.DEFINE_integer("image_size", 33, "The size of image to use [33]")
flags.DEFINE_integer("label_size", 21, "The size of label to produce [21]")
flags.DEFINE_integer("num_F1", 64, "The number of feature map for first layer [64]")
flags.DEFINE_integer("num_F2", 32, "The number of feature map for second layer [32]")
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_integer("scale", 3, "The size of scale factor for preprocessing input image [3]")
flags.DEFINE_integer("stride", 21, "The size of stride to apply input image. 14 for train, 21 for test [14]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("h5_dir", "h5_dir", "Name of h5 directory [h5_dir]")
flags.DEFINE_string("save_dir", "save_dir", "Name of saving result directory [save_dir]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [True]")

# print見やすくするだけ
pp = pprint.PrettyPrinter()


def graph_output(history):

    now = datetime.datetime.now()
    savename = './' + FLAGS.checkpoint_dir + '/psnr_loss_' + now.strftime('%Y%m%d_%H%M%S') + '.png'

    psnr_loss_fig = plt.figure()

    # PSNRグラフ
    psnr = psnr_loss_fig.add_subplot(111)
    psnr.plot(history.history['psnr'], color='blue', label='PSNR')

    # 損失値グラフ
    loss = psnr.twinx()
    loss.plot(history.history['loss'], color='red', label='LOSS')

    h1, l1 = psnr.get_legend_handles_labels()
    h2, l2 = loss.get_legend_handles_labels()
    psnr.legend(h1 + h2, l1 + l2, loc='center right')

    psnr_loss_fig.savefig(savename)


def main(_):
    # 設定情報出力
    pp.pprint(FLAGS.flag_values_dict())

    # 学習用のチェックポイントフォルダ作成
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    # .h5のサブ画像格納用のフォルダ作成
    if not os.path.exists(FLAGS.h5_dir):
        os.makedirs(FLAGS.h5_dir)

    # 結果保存用のフォルダ作成
    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)

    """
    srcnn_archtecture = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(FLAGS.image_size, FLAGS.image_size, FLAGS.c_dim)),
            layers.Conv2D(FLAGS.num_F1, (9, 9), strides=(1, 1), padding="same", activation='relu'),
            layers.Conv2D(FLAGS.num_F2, (1, 1), strides=(1, 1), padding="same", activation='relu'),
            layers.Conv2D(1, (5, 5), strides=(1, 1), padding="same", activation='relu'),
        ],
        name="Srcnn915",
    )
    """
    """
    srcnn = SRCNN(srcnn_archtecture,
                  image_size=FLAGS.image_size,
                  label_size=FLAGS.label_size,
                  batch_size=FLAGS.batch_size,
                  c_dim=FLAGS.c_dim,
                  checkpoint_dir=FLAGS.checkpoint_dir,
                  sample_dir=FLAGS.sample_dir)

    """
    srcnn = SRCNN(image_size=FLAGS.image_size,
                  label_size=FLAGS.label_size,
                  batch_size=FLAGS.batch_size,
                  c_dim=FLAGS.c_dim,
                  checkpoint_dir=FLAGS.checkpoint_dir)

    if FLAGS.is_train:
        his, model = srcnn.train_step(FLAGS)
        graph_output(his)
    else:
        srcnn.test_all(FLAGS)


if __name__ == '__main__':
    app.run(main)


