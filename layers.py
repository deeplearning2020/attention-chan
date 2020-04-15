import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, SeparableConv2D, Conv2DTranspose,
            BatchNormalization, AveragePooling2D, MaxPooling2D, LeakyReLU, Add, Activation)



class AttentionBlock(object):
    def __init__(self, filters):

        super(AttentionBlock, self).__init__()
        self.filters = filters

    def __call__(self, x):

        maxpool = MaxPooling2D(pool_size = 2,strides = 1, padding = 'same')(x)
        avgpool = AveragePooling2D(pool_size = 2, strides = 1,padding = 'same')(x)      
        x = tf.multiply(maxpool, avgpool)

        g1 = Conv2D(self.filters, kernel_size = 1)(x) 
        g1 = BatchNormalization()(g1)

        x1 = Conv2D(self.filters, kernel_size = 1)(x) 
        x1 = BatchNormalization()(x1)

        g2 = Conv2D(self.filters, kernel_size = 1)(x) 
        g2 = BatchNormalization()(g2)

        x2 = Conv2D(self.filters, kernel_size = 1)(x) 
        x2 = BatchNormalization()(x2)

        g3 = Conv2D(self.filters, kernel_size = 1)(x) 
        g3 = BatchNormalization()(g3)

        
        g1_x1 = Add()([g1, x1, g2, x2, g3])
        psi = LeakyReLU()(g1_x1)

        psi = Conv2D(1,kernel_size = 1)(psi) 
        psi = BatchNormalization()(psi)
        psi = Activation('tanh')(psi)

        x = tf.multiply(x,psi)
        return x


class DepthwiseSeparableConv_Block(object):

    def __init__(self, filters, kernelSize, strides = 1):

        self.filters = filters
        self.kernelSize = kernelSize
        self.strides = strides

    def __call__(self, x, training = None):

        x = SeparableConv2D(self.filters, self.kernelSize, strides = self.strides, padding = 'same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        return x


class Deconv_Block(object):

    def __init__(self, filters, kernelSize, strides = 1):

        self.filters = filters
        self.kernelSize = kernelSize
        self.strides = strides

    def __call__(self, x, training = None):

        x = Conv2DTranspose(self.filters, self.kernelSize, strides = self.strides, padding = 'same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        return x

class Conv_Block(object):

    def __init__(self, filters, kernelSize, strides = 1):
        self.filters = filters
        self.kernelSize = kernelSize
        self.strides = strides

    def __call__(self, x, training = None):

        x = Conv2D(self.filters, self.kernelSize, strides = self.strides, padding = 'same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        return x

