import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Add, Activation, Multiply
from tensorflow.keras.layers import BatchNormalization



class Attention(object):
    def __init__(self, filters):
        super(Attention, self).__init__()
        self.filters = filters
    def __call__(self, x):
        maxpool = MaxPooling2D(pool_size = 2,strides = 1, padding = 'same')(x)
        avgpool = AveragePooling2D(pool_size = 2, strides = 1,padding = 'same')(x)
        x = tf.add(maxpool,avgpool)
        g1 = Conv2D(self.filters, kernel_size = 1)(x) 
        g1 = BatchNormalization()(g1)
        x1 = Conv2D(self.filters, kernel_size = 1)(x) 
        x1 = BatchNormalization()(x1)
        g2 = Conv2D(self.filters, kernel_size = 1)(x) 
        g2 = BatchNormalization()(g1)
        x2 = Conv2D(self.filters, kernel_size = 1)(x) 
        x2 = BatchNormalization()(x1)
        g3 = Conv2D(self.filters, kernel_size = 1)(x) 
        g3 = BatchNormalization()(g1)
        x3 = Conv2D(self.filters, kernel_size = 1)(x) 
        x3 = BatchNormalization()(x1)
        
        g1_x1 = Add()([g1, x1,g2, x2,g3, x3])
        psi = Activation('relu')(g1_x1)
        psi = Conv2D(1,kernel_size = 1)(psi) 
        psi = BatchNormalization()(psi)
        psi = Activation('tanh')(psi)
        x = Multiply()([x,psi])
        return x



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

class SpatialAttention(object):

    def __init__(self, filters):

        super(SpatialAttention, self).__init__()
        self.filters = filters

    def __call__(self, x):

        skip_conn = tf.identity(x, name='identity')

        #maxpool = MaxPooling2D((2, 2), padding='valid')(x)

        #avgpool = AveragePooling2D((2, 2), padding='valid')(x)

        #x = tf.add(maxpool, avgpool)
        avgpool = AveragePooling2D(2, strides = 1, padding = 'same')(x)
        maxpool = MaxPooling2D(2, strides = 1, padding = 'same')(x)
        x = tf.multiply(avgpool, maxpool)

        x = Conv2D(self.filters, kernel_size = 1, activation = 'relu')(x)

        x = Conv2D(self.filters, kernel_size = 1, activation = 'sigmoid')(x)

        x = tf.multiply(skip_conn, x)

        return x
