import tensorflow as tf

from models.CRU import CRU
from models.Common import Downsample, Conv


class SFL(tf.keras.Model):

    def __init__(self, filters, size=3, apply_batchnorm=True):
        super(SFL, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        # depth map
        self.cru1 = CRU(filters, size, stride=1)
        self.conv1 = Conv(2, size, activation=False, apply_batchnorm=False)

        # class
        self.conv2 = Downsample(filters * 1, size)
        self.conv3 = Downsample(filters * 1, size)
        self.conv4 = Downsample(filters * 2, size)
        self.conv5 = Downsample(filters * 4, 4, padding='VALID')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = Dense(256)
        self.fc2 = Dense(1, activation=False, apply_batchnorm=False)

        self.dropout = tf.keras.layers.Dropout(0.3)

    def call(self, x, training):
        # depth map branch
        xd = self.cru1(x)
        xd = self.conv1(xd)
        dmap = tf.nn.sigmoid(xd)
        # class branch
        x = self.conv2(x)  # 16*16*32
        x = self.conv3(x)  # 8*8*64
        x = self.conv4(x)  # 4*4*128
        x = self.conv5(x)  # 1*1*256
        x = self.flatten(x)
        x = self.dropout(x, training=training)
        x = self.fc1(x)
        cls = self.fc2(x)
        return dmap, cls


class Dense(tf.keras.Model):
    def __init__(self, filters, activation=True, apply_batchnorm=True, apply_dropout=False):
        super(Dense, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        self.activation = activation
        self.apply_dropout = apply_dropout
        initializer = tf.random_normal_initializer(0., 0.02)
        filters = int(filters)
        self.dense = tf.keras.layers.Dense(filters,
                                           kernel_initializer=initializer,
                                           use_bias=False)
        if self.apply_batchnorm:
            self.batchnorm = tf.keras.layers.BatchNormalization()
        if self.apply_dropout:
            self.dropout = tf.keras.layers.Dropout(0.3)

    def call(self, x, training):
        x = self.dense(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x, training=training)
        if self.activation:
            x = tf.nn.leaky_relu(x)
        if self.apply_dropout:
            x = self.dropout(x, training=training)
        return x
