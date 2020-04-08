from tensorflow.python.keras.layers import (InputLayer, Conv2D, Conv2DTranspose,
            BatchNormalization, LeakyReLU, MaxPool2D, UpSampling2D,
            Reshape, GlobalAveragePooling2D, GaussianNoise)
from tensorflow.python.keras.models import Model
from layers import SelfAttention

class ConvATT(object):

    def __init__(self, filters, kernelSize, strides = 1):
        self.filters = filters
        self.kernelSize = kernelSize
        self.strides = strides

    def __call__(self, x, training = None):

        x = Conv2D(self.filters, self.kernelSize, strides = self.strides, padding = 'same')(x)
        x = SelfAttention(ch = self.filters)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        return x



