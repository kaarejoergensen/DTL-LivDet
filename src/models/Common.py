import tensorflow as tf


class Conv(tf.keras.Model):
    def __init__(self, filters, size, stride=1, activation=True, padding='SAME', apply_batchnorm=True):
        super(Conv, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        self.activation = activation
        initializer = tf.random_normal_initializer(0., 0.02)
        filters = int(filters)
        self.conv1 = tf.keras.layers.Conv2D(filters,
                                            (size, size),
                                            strides=stride,
                                            padding=padding,
                                            kernel_initializer=initializer,
                                            use_bias=False)
        if self.apply_batchnorm:
            self.batchnorm = tf.keras.layers.BatchNormalization()

    def call(self, x, training):
        x = self.conv1(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x, training=training)
        if self.activation:
            x = tf.nn.leaky_relu(x)
        return x


class Downsample(tf.keras.Model):
    def __init__(self, filters, size, padding='SAME', apply_batchnorm=True):
        super(Downsample, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        initializer = tf.random_normal_initializer(0., 0.02)
        filters = int(filters)
        self.conv1 = tf.keras.layers.Conv2D(filters, (size, size), strides=2, padding=padding,
                                            kernel_initializer=initializer, use_bias=False)
        if self.apply_batchnorm:
            self.batchnorm = tf.keras.layers.BatchNormalization()

    def call(self, x, training):
        x = self.conv1(x)
        if self.apply_batchnorm:
            x = self.batchnorm(x, training=training)
        x = tf.nn.leaky_relu(x)
        return x
