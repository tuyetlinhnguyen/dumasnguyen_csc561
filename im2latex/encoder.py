import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# image readers
from PIL import Image
import cv2

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, LSTM, Softmax
from keras.preprocessing.image import ImageDataGenerator

from global_ import IMAGE_HEIGHT, IMAGE_WIDTH

### MODEL DEFINITION
### currently VGG16 model
def cnn_encoder_decoder():

    ### CNN layers ###
    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH,3))) 
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu')) 
    model.add(MaxPooling2D(pool_size=(2,2), strides=2))

    filters_convs = [(128, 2), (256, 2), (512, 2)]#, (512,3)] 
    for n_filters, n_convs in filters_convs:
        for _ in np.arange(n_convs):
            model.add(Conv2D(filters=n_filters, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=2))
    
    model.add(Flatten())
    model.add(Dense(4096, activation = 'relu')) 
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation = 'relu')) 
    model.add(Dropout(0.5)) 
    model.add(Dense(1000,activation = 'softmax'))

    opt = keras.SGD(lr = 0.01)

    ### CNN layers ###

    model.compile(loss = keras.categorical_crossentropy, optimizer = opt, metrics = ['accuracy'])

    return model


def rnn_decoder():
    return

