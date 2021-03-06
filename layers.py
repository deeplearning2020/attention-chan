import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    SeparableConv2D,
    Conv2DTranspose,
    BatchNormalization,
    Reshape,
    Dense,
    AveragePooling2D,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    MaxPooling2D,
    LeakyReLU,
    PReLU,
    Add,
    Activation)
from tensorflow.keras.initializers import RandomNormal


class AttentionBlock(object):
    def __init__(self, filters):

        super(AttentionBlock, self).__init__()
        self.filters = filters
        #self.init = RandomNormal()

    def __call__(self, x):

        g1 = Conv2D(self.filters, kernel_size=1, padding='same')(x)
        g1 = GlobalAveragePooling2D()(g1)
        g1 = Reshape((1, 1, self.filters))(g1)
        g1 = Conv2D(self.filters, kernel_size=3, padding='same')(g1)

        x1 = Conv2D(self.filters, kernel_size=1, padding='same')(x)
        x1 = GlobalMaxPooling2D()(x1)
        x1 = Reshape((1, 1, self.filters))(x1)
        x1 = Conv2D(self.filters, kernel_size=3, padding='same')(x1)

        p1 = Conv2D(self.filters, kernel_size=1, padding='same')(x)
        p1 = GlobalAveragePooling2D()(p1)
        p1 = Reshape((1, 1, self.filters))(p1)
        p1 = Conv2D(self.filters, kernel_size=3, padding='same')(p1)

        t1 = Conv2D(self.filters, kernel_size=1, padding='same')(x)
        t1 = GlobalAveragePooling2D()(t1)
        t1 = Reshape((1, 1, self.filters))(t1)
        t1 = Conv2D(self.filters, kernel_size=3, padding='same')(t1)

        psi = Add()([g1, x1, p1, t1, x])
        #psi = LeakyReLU()(psi)
        psi = GlobalAveragePooling2D()(psi)
        #psi = Dense(self.filters)(psi)
        psi = Reshape((1, 1, self.filters))(psi)
        psi = Activation('softmax')(psi)

        g2 = Conv2D(self.filters, kernel_size=1, padding='same')(psi)

        x2 = Conv2D(self.filters, kernel_size=1, padding='same')(psi)

        p2 = Conv2D(self.filters, kernel_size=1, padding='same')(psi)

        psi = Add()([g2, x2, x, p2, psi])
        psi = LeakyReLU(alpha = 0.7)(psi)
        psi = Conv2D(1, kernel_size=1, padding='same')(psi)
        #psi = BatchNormalization()(psi)
        psi = Activation('sigmoid')(psi)
        #psi = MaxPooling2D(pool_size=(2, 2), padding='same')(psi)
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
        x = LeakyReLU(alpha = 0.7)(x)
        #x = BatchNormalization()(x)
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
