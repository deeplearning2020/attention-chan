import os, math
import numpy as np
import cv2, skimage
from PIL import Image
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import Input, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, GaussianNoise, LeakyReLU, MaxPooling2D, AveragePooling2D
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
from util import Conv_2D, simple_conv, Deconv
from layers import SelfAttention
from attention_layer import ChannelAttention, SpatialAttention
from util import Resnet_block


def model(inputShape):
    input_img = Input(shape=(inputShape))
    x = Conv_2D(128, 5, strides = 1)(input_img)
    x = Conv_2D(128, 3, strides = 1)(x)
    x = Conv_2D(64, 1, strides = 1)(x)
    #x = SpatialAttention(64)(x)
    x = Conv_2D(64, 5, strides = 1)(x)
    x = Conv_2D(64, 3, strides = 1)(x)
    x = Conv_2D(32, 1, strides = 1)(x)
    #x = SpatialAttention(32)(x)
    x = Conv_2D(32, 5, strides = 1)(x)
    x = Conv_2D(32, 3, strides = 1)(x)
    x = Conv_2D(16, 1, strides = 1)(x)
    #x = SpatialAttention(16)(x)
    x = Conv_2D(16, 5, strides = 1)(x)
    x = Conv_2D(16, 3, strides = 1)(x)
    x = Conv_2D(8, 3, strides = 1)(x)
    x = Conv_2D(3, 3, strides = 1)(x)
    avgpool = AveragePooling2D(2, strides = 1, padding = 'same')(x)
    maxpool = MaxPooling2D(2, strides = 1, padding = 'same')(x)
    x = tf.multiply(avgpool, maxpool)
    model = Model(input_img, x)
    return model




def main():

    inputShape = (None, None, 3)
    batchSize = 2

    hr_image = load_img(os.path.join(os.getcwd(),'hr_image','HR.bmp'))
            #target_size = inputShape[:-1]) ## loading the high-resolution image
    hr_image = np.array(hr_image, dtype = np.float32) * (2/255) - 1
    hr_image = np.array([hr_image]*batchSize) ## creating fake batches


    lr_image = load_img(os.path.join(os.getcwd(),'lr_image','LR.bmp'))
            #target_size = inputShape[:-1]) ## loading the low-resolution image
    lr_image = np.array(lr_image, dtype = np.float32) * (2/255) - 1
    lr_image = np.array([lr_image]*batchSize)

    nn = model(inputShape)
    print(nn.summary())
    optimizer = Adam(lr=1e-2, epsilon = 1e-8, beta_1 = .9, beta_2 = .999)
    nn.compile(optimizer = optimizer, loss = 'mse')
    
    es = EarlyStopping(monitor = 'loss' , mode = 'min', verbose = 1, 
            patience = 25) ## early stopping to prevent overfitting

    history = nn.fit(lr_image, hr_image,
                epochs = 1000,
                batch_size = batchSize, callbacks = [es])

    """ reconstrucing high-resolution image from the low-resolution image """
    pred = nn.predict(lr_image)
    pred = np.uint8((pred + 1)* 255/2)
    pred = Image.fromarray(pred[0])
    pred.save("reconstructed_HR_image.png")

if __name__ == "__main__":
    main()
