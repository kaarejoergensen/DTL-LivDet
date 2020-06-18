import tensorflow as tf


class CRU(tf.keras.Model):
    COUNT = 4

    def __init__(self, filters, size=3, stride=2, apply_batchnorm=True):
        super(CRU, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        self.stride = stride
        initializer = tf.random_normal_initializer(0., 0.02)
        filters = int(filters)

        for count in range(self.COUNT):
            conv = tf.keras.layers.Conv2D(filters, (size, size), strides=1, padding='SAME',
                                          kernel_initializer=initializer, use_bias=False)
            batchnorm = tf.keras.layers.BatchNormalization()
            setattr(self, "conv_{}".format(count), conv)
            setattr(self, "batchnorm_{}".format(count), batchnorm)

    def call(self, x, training):
        original = x
        for count in range(self.COUNT):
            x = getattr(self, "conv_{}".format(count))(x)
            x = getattr(self, "batchnorm_{}".format(count))(x, training=training)
            # Skip connection in every other layer
            if count % 2 != 0:
                x = original + x
            x = tf.nn.leaky_relu(x)

        if self.stride > 1:
            x = tf.nn.max_pool(x, 3, 2, padding='SAME')
        return x
