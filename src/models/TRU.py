import tensorflow as tf

from models.Common import Downsample


class TRU(tf.keras.Model):
    def __init__(self, filters, idx, alpha=1e-3, beta=1e-4, size=3, apply_batchnorm=True):
        super(TRU, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        # variables
        self.conv1 = Downsample(filters, size)
        self.conv2 = Downsample(filters, size)
        self.conv3 = Downsample(filters, size)
        self.flatten = tf.keras.layers.Flatten()
        self.project = Linear(idx, alpha, beta, input_dim=2048)

    def call(self, x, mask, training):
        # Downsampling
        x_small = self.conv1(x, training=training)
        depth = 0
        if x_small.shape[1] > 16:
            x_small = self.conv2(x_small, training=training)
            depth += 1
            if x_small.shape[1] > 16:
                x_small = self.conv3(x_small, training=training)
                depth += 1
        x_small_shape = x_small.shape
        x_flatten = self.flatten(tf.nn.avg_pool(x_small, ksize=3, strides=2, padding='SAME'))

        # PCA Projection
        route_value, route_loss, uniq_loss = self.project(x_flatten, mask, training=training)

        # Generate the splitting mask
        mask_l = mask * tf.cast(tf.greater_equal(route_value, tf.constant(0.)), tf.float32)
        mask_r = mask * tf.cast(tf.less(route_value, tf.constant(0.)), tf.float32)

        return [mask_l, mask_r], route_value, [route_loss, uniq_loss]


class Linear(tf.keras.layers.Layer):
    def __init__(self, idx, alpha, beta, input_dim=32):
        super(Linear, self).__init__()
        initializer = tf.random_normal_initializer(0., 0.02)
        initializer0 = tf.zeros_initializer()
        self.v = tf.Variable(initial_value=initializer(shape=(1, input_dim), dtype='float32'),
                             trainable=True, name='tru/v/' + idx)
        self.mu = tf.Variable(initial_value=initializer0(shape=(1, input_dim), dtype='float32'),
                              trainable=True, name='tru/mu/' + idx)
        # training hyper-parameters
        self.alpha = alpha
        self.beta = beta
        # mean, eigenvalue and trace for each mini-batch
        self.mu_of_visit = 0.
        self.eigenvalue = 0.
        self.trace = 0.

    def call(self, x, mask, training):
        norm_v = self.v / (tf.norm(self.v) + 1e-8)
        norm_v_t = tf.transpose(norm_v, [1, 0])
        num_of_visit = tf.reduce_sum(mask)

        if training and num_of_visit > 1:
            # use only the visiting samples
            # index = tf.where(tf.greater(mask, tf.constant(0.)))
            # index_not = tf.where(tf.equal(mask, tf.constant(0.)))
            # x_sub = tf.gather_nd(x, index) - tf.stop_gradient(self.mu)
            # x_not = tf.gather_nd(x, index_not) - tf.stop_gradient(self.mu)
            # x_sub_t = tf.transpose(x_sub, [1, 0])
            index = tf.where(tf.greater(mask[:, 0], tf.constant(0.)))
            index_not = tf.where(tf.equal(mask[:, 0], tf.constant(0.)))
            x_sub = tf.gather_nd(x, index) - tf.stop_gradient(self.mu)
            x_not = tf.gather_nd(x, index_not)
            x_sub_t = tf.transpose(x_sub, [1, 0])

            # compute the covariance matrix, eigenvalue, and the trace
            covar = tf.matmul(x_sub_t, x_sub) / num_of_visit
            eigenvalue = tf.reshape(tf.matmul(tf.matmul(norm_v, covar), norm_v_t), [])
            trace = tf.linalg.trace(covar)
            # compute the route loss
            # print(tf.exp(-self.alpha * eigenvalue), self.beta * trace)
            route_loss = tf.exp(-self.alpha * eigenvalue) + self.beta * trace
            uniq_loss = -tf.reduce_mean(tf.square(tf.matmul(x_sub, norm_v_t))) + \
                        tf.reduce_mean(tf.square(tf.matmul(x_not, norm_v_t)))
            # compute mean and response for this batch
            self.mu_of_visit = tf.reduce_mean(x_sub, axis=0, keepdims=True)
            self.eigenvalue = eigenvalue
            self.trace = trace
            x -= tf.stop_gradient(self.mu_of_visit)
            route_value = tf.matmul(x, norm_v_t)
        else:
            self.mu_of_visit = self.mu
            self.eigenvalue = 0.
            self.trace = 0.
            x -= self.mu
            route_value = tf.matmul(x, norm_v_t)
            route_loss = 0.
            uniq_loss = 0.

        return route_value, route_loss, uniq_loss
