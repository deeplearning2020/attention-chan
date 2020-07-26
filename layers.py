import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    SeparableConv2D,
    Conv2DTranspose,
    BatchNormalization,
    Reshape,
    AveragePooling2D,
    GlobalAveragePooling2D,
    MaxPooling2D,
    LeakyReLU,
    Add,
    Activation)
from tensorflow.keras.initializers import RandomNormal


class AttentionBlock(object):
    def __init__(self, filters):

        super(AttentionBlock, self).__init__()
        self.filters = filters
        #self.init = RandomNormal()

    def __call__(self, x):

        #self.init = RandomNormal()
        #maxpool = MaxPooling2D(pool_size = 2,strides = 1, padding = 'same')(x)
        #avgpool = AveragePooling2D(pool_size = 2, strides = 1,padding = 'same')(x)
        #x = tf.multiply(maxpool, avgpool)

        g1 = Conv2D(self.filters, kernel_size=3, padding='same')(x)

        x1 = Conv2D(self.filters, kernel_size=5, padding='same')(x)

        p1 = Conv2D(self.filters, kernel_size=7, padding='same')(x)

        #x2 = Conv2D(self.filters, kernel_size = 1)(x)

        #g3 = Conv2D(self.filters, kernel_size = 1)(x)

        #x3 = Conv2D(self.filters, kernel_size = 1)(x)

        psi = Add()([g1, x1, p1])
        psi = LeakyReLU()(psi)
        psi = GlobalAveragePooling2D()(psi)
        g2 = Conv2D(self.filters, kernel_size=3, padding='same')(psi)

        x2 = Conv2D(self.filters, kernel_size=5, padding='same')(psi)

        p2 = Conv2D(self.filters, kernel_size=7, padding='same')(psi)
        psi = Add()([g2, x2, p2])
        psi = LeakyReLU()(psi)
        psi = Reshape((1, 1, self.filters))(psi)
        psi = Activation('softmax')(psi)

        psi = Conv2D(1, kernel_size=1, padding='same')(psi)
        psi = BatchNormalization()(psi)
        psi = Activation('sigmoid')(psi)
        psi = MaxPooling2D(pool_size=(2, 2), padding='same')(psi)
        x = tf.multiply(x, psi)
        return x


class DepthwiseSeparableConv_Block(object):

    def __init__(self, filters, kernelSize, strides=1):

        self.filters = filters
        self.kernelSize = kernelSize
        self.strides = strides
        #self.init = RandomNormal()

    def __call__(self, x, training=None):

        #self.init = RandomNormal()
        x = Conv2D(
            self.filters,
            kernel_size=self.kernelSize,
            strides=self.strides,
            padding='same')(x)
        #x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        return x


class Deconv_Block(object):

    def __init__(self, filters, kernelSize, strides=1):

        self.filters = filters
        self.kernelSize = kernelSize
        self.strides = strides

    def __call__(self, x, training=None):

        x = Conv2DTranspose(
            self.filters,
            self.kernelSize,
            strides=self.strides,
            padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        return x


class Conv_Block(object):

    def __init__(self, filters, kernelSize, strides=1):
        self.filters = filters
        self.kernelSize = kernelSize
        self.strides = strides

    def __call__(self, x, training=None):

        x = Conv2D(
            self.filters,
            self.kernelSize,
            strides=self.strides,
            padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        return x
