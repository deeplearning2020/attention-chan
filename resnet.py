from tensorflow.keras.layers import Conv2D, BatchNormalization, Add, Activation, LeakyReLU

def res_net_block(input_data, filters, conv_size):
    x = Conv2D(filters, conv_size,padding='same')(input_data)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, conv_size, activation=None, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, input_data])
    x = LeakyReLU()(x)
    return x
