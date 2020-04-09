import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Conv2DTranspose,
            BatchNormalization, LeakyReLU)
from tensorflow.keras.models import Model
from layers import SelfAttention

class Resnet_block(object):
  def __init__(self, filters, kernelSize):
    self.filters = filters
    self.kernelSize = kernelSize

    self.conv2a = Conv2D(filters, (1, 1))
    self.bn2a = BatchNormalization()

    self.conv2b = Conv2D(filters, kernelSize, padding = 'same')
    self.bn2b = BatchNormalization()

    self.conv2c = Conv2D(filters, (1, 1))
    self.bn2c = BatchNormalization()

  def __call__(self, input_tensor, training = False):
    x = self.conv2a(input_tensor)
    x = self.bn2a(x, training = training)
    x = tf.nn.relu(x)

    x = self.conv2b(x)
    x = self.bn2b(x, training = training)
    x = tf.nn.relu(x)

    x = self.conv2c(x)
    x = self.bn2c(x, training = training)

    x += input_tensor
    return tf.nn.relu(x)



class ConvATT(object):

    def __init__(self, filters, kernelSize, strides = 2):
        self.filters = filters
        self.kernelSize = kernelSize
        self.strides = strides

    def __call__(self, x, training = None):

        x = Conv2D(self.filters, self.kernelSize, strides = self.strides, padding = 'same')(x)
        x = SelfAttention(ch = self.filters)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        return x

class Conv_2D(object):

    def __init__(self, filters, kernelSize, strides = 1):
        self.filters = filters
        self.kernelSize = kernelSize
        self.strides = strides

    def __call__(self, x, training = None):

        x = Conv2D(self.filters, self.kernelSize, strides = self.strides, padding = 'same')(x)
        #x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        return x

class Deconv(object):

    def __init__(self, filters, kernelSize, strides = 2):
        self.filters = filters
        self.kernelSize = kernelSize
        self.strides = strides

    def __call__(self, x, training = None):

        x = Conv2DTranspose(self.filters, self.kernelSize, strides = self.strides, padding = 'same')(x)
        #x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        return x



