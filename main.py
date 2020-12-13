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

# 33*33の入力が9*1*5の畳み込みで画像周り6pix分小さくなるため、ラベルは21*21となる
# 学習時は"is_train"をFalseにする
FLAGS = flags.FLAGS
flags.DEFINE_integer("epoch", 500, "Number of epoch [500]")
flags.DEFINE_integer("batch_size", 128, "The size of batch images [128]")
flags.DEFINE_integer("image_size", 33, "The size of image to use [33]")
flags.DEFINE_integer("label_size", 21, "The size of label to produce [21]")
flags.DEFINE_integer("num_F1", 64, "The number of feature map for first layer [64]")
flags.DEFINE_integer("num_F2", 32, "The number of feature map for second layer [32]")
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_integer("scale", 3, "The size of scale factor for preprocessing input image [3]")
flags.DEFINE_integer("stride", 14, "The size of stride to apply input image [14]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("sample_dir", "sample", "Name of sample directory [sample]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")

# print見やすくするだけ
pp = pprint.PrettyPrinter()


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


def main(_):
    # 設定情報出力
    pp.pprint(FLAGS.flag_values_dict())

    # 学習用のチェックポイントフォルダ作成
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

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
                  checkpoint_dir=FLAGS.checkpoint_dir,
                  sample_dir=FLAGS.sample_dir)

    his, model = srcnn.train_step(FLAGS)
    
    graph_output(his)


if __name__ == '__main__':
    app.run(main)


