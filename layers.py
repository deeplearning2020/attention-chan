import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, SeparableConv2D, Conv2DTranspose,
            BatchNormalization, AveragePooling2D, MaxPooling2D, LeakyReLU, Add, Activation)
from tensorflow.keras.initializers import RandomNormal



class AttentionBlock(object):
    def __init__(self, filters):

        super(AttentionBlock, self).__init__()
        self.filters = filters
        self.init = RandomNormal()
    def __call__(self, x):

        #self.init = RandomNormal()
        maxpool = MaxPooling2D(pool_size = 2,strides = 1, padding = 'same')(x)
        avgpool = AveragePooling2D(pool_size = 2, strides = 1,padding = 'same')(x)      
        x = tf.multiply(maxpool, avgpool)

        g1 = SeparableConv2D(self.filters, kernel_initializer = self.init, kernel_size = 1)(x) 

        x1 = SeparableConv2D(self.filters, kernel_initializer = self.init, kernel_size = 1)(x) 

        g2 = Conv2D(self.filters, kernel_initializer = self.init, kernel_size = 1)(x) 

        x2 = Conv2D(self.filters, kernel_initializer = self.init, kernel_size = 1)(x) 

        g3 = Conv2D(self.filters, kernel_initializer = self.init, kernel_size = 1)(x) 

        x3 = Conv2D(self.filters, kernel_initializer = self.init, kernel_size = 1)(x)

        
        g1_x1 = Add()([g1, x1, g2, x2, g3, x3])
        psi = LeakyReLU()(g1_x1)

        psi = Conv2D(1, kernel_initializer = self.init, kernel_size = 1)(psi) 
        psi = BatchNormalization()(psi)
        psi = Activation('tanh')(psi)

        x = tf.multiply(x,psi)
        return x


class DepthwiseSeparableConv_Block(object):

    def __init__(self, filters, kernelSize, strides = 1):

        self.filters = filters
        self.kernelSize = kernelSize
        self.strides = strides
        self.init = RandomNormal()

    def __call__(self, x, training = None):

        #self.init = RandomNormal()
        x = SeparableConv2D(self.filters, kernel_size = self.kernelSize, kernel_initializer = self.init, strides = self.strides, padding = 'same')(x)
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

