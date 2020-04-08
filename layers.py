import tensorflow as tf
from tensorflow.python.keras.layers import Input, Layer, InputSpec, Flatten
from tensorflow.python.keras.initializers import RandomNormal

class SelfAttention(Layer):

    def __init__(self, ch, **kwargs):

        super(SelfAttention, self).__init__(**kwargs)
        self.channels = ch
        self.filters_f_g = self.channels // 8
        self.filters_h = self.channels

    def build(self, input_shape):

        self.init = RandomNormal(stddev = 0.04)
        kernel_shape_f_g = (1, 1) + (self.channels, self.filters_f_g)
        kernel_shape_h = (1, 1) + (self.channels, self.filters_h)
        self.gamma = self.add_weight(name = 'gamma', shape = [1], initializer = self.init, trainable=True)

        self.kernel_f = self.add_weight(shape = kernel_shape_f_g,
                                        initializer = self.init,
                                        name = 'kernel_f',
                                        trainable = True)
        self.kernel_g = self.add_weight(shape =kernel_shape_f_g,
                                        initializer = self.init,
                                        name = 'kernel_g',
                                        trainable = True)
        self.kernel_h = self.add_weight(shape=kernel_shape_h,
                                        initializer = self.init,
                                        name = 'kernel_h',
                                        trainable = True)

        super(SelfAttention, self).build(input_shape)

        self.input_spec = InputSpec(ndim = 4,
                                    axes = {3: input_shape[-1]})
        self.built = True

    def call(self, x):
        
        def hw_flatten(x):
            x_shape = tf.shape(x)
            return tf.reshape(x, [x_shape[0], -1, x_shape[-1]]) 

        f = tf.nn.conv2d(x,
                     filter = self.kernel_f,
                     strides = (1, 1), padding = 'SAME')

        g = tf.nn.conv2d(x,
                     filter = self.kernel_g,
                     strides = (1, 1), padding = 'SAME')

        h = tf.nn.conv2d(x,
                     filter = self.kernel_h,
                     strides = (1, 1), padding = 'SAME')

        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b = True)
        beta = tf.nn.softmax(s, axis = -1)
        o = tf.matmul(beta, hw_flatten(h))
        o = tf.reshape(o, shape = tf.shape(x))
        x = self.gamma * o + x

        return x

    def compute_output_shape(self, input_shape):
        return input_shape
