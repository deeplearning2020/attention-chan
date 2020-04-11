import tensorflow as tf
from tensorflow.keras.layers import Conv2D

class ChannelAttention(object):

    def __init__(self, filters, reduction):

        super(ChannelAttention, self).__init__()
        self.filters = filters
        self.reduction = reduction

    def __call__(self, x):

        def adaptive_global_average_pool_2d(x):

            c = x.get_shape()[-1]

            return tf.reshape(tf.reduce_mean(x, axis = [1, 2]), (-1, 1, 1, c))

        skip_conn = tf.identity(x, name='identity')

        x = adaptive_global_average_pool_2d(x)

        x = Conv2D(self.filters//self.reduction, kernel_size = 1, activation = 'relu')(x)

        x = Conv2D(self.filters, kernel_size = 1, activation = 'sigmoid')(x)

        x = tf.multiply(skip_conn, x)

        return x
