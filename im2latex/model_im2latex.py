import os
import sys

import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# image readers
from PIL import Image
import cv2

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

import wandb
# start a new wandb run to track this script
""" 
wandb.init(
    # set the wandb project where this run will be logged
    project="csc561_model_test",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "inkml",
    "epochs": 10,
    }
)
 """

# paths for images before and after conversion (inkml2img)
INPUT_IMG_PATH = '/Users/tuyet/Documents/S23/CSC 561/Final Project/Final/dumasnguyen_csc561/data/batch_2/background_images/'
JSON_FILE = '/Users/tuyet/Documents/S23/CSC 561/Final Project/Final/dumasnguyen_csc561/data/batch_2/JSON/kaggle_data_2.json'

# INPUT IMAGE STANDARD SIZE
IMG_HEIGHT = 100
IMG_WIDTH = 275
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# Function provided at https://www.kaggle.com/code/aidapearson/eda-starter-notebook/input
def create_data_frame(raw_data):
    """
    Create a Pandas DataFrame from data json file

    Parameters
    ----------
    raw_data : list
        A list that contains all the image information

    Returns
    ----------
    df: DataFrame
        A Pandas DataFrame for running the analysis
    all_latex_lst: list
        A list for all the tokens, used for creating the token distribution
    """
    data = {}
    data["filename"] = []
    data['latex'] = []
    data['seq_len'] = []
    data['font'] = []
    data['image_ratio'] = []
    data['image_width'] = []
    data['image_height'] = []
    all_latex_lst = []

    for image in raw_data:
        data["filename"].append(image["filename"])
        data['latex'].append(image['image_data']['full_latex_chars'])
        data['seq_len'].append(len(image['image_data']['full_latex_chars']))
        data['font'].append(image['font'])
        data['image_ratio'].append(round(image['image_data']['width'] / image['image_data']['height'],1))
        data['image_width'].append(image['image_data']['width'])
        data['image_height'].append(image['image_data']['height'])
        all_latex_lst = all_latex_lst + image['image_data']['full_latex_chars']
    df = pd.DataFrame.from_dict(data)
    return df, all_latex_lst

# Load data into a Pandas DataFrame and store all the tokens into a list.
with open(JSON_FILE) as f:
    raw_data = json.load(f)

data_df, latex_tokens = create_data_frame(raw_data)

# all_images = data_df['filename'].values.tolist()
# all_labels = data_df['latex'].values.tolist()

# # split into train/validation/test - 80/10/10 split
# split = [int(len(all_images)*0.8), int(len(all_labels)*0.9)]
# train_images, val_images, test_images = np.split(all_images, split)
# train_labels, val_labels, test_labels = np.split(all_labels, split)

split = [int(len(data_df)*0.8), int(len(data_df)*0.9)]
train_df, val_df, test_df = np.split(data_df, split)

# min height & width: 103 & 275 (respectively)
# rescale all images to 100x275 (275x100?)
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(train_df, directory=INPUT_IMG_PATH, x_col='filename', y_col='latex',
                                                    target_size=IMG_SIZE, batch_size=32, class_mode='categorical')
val_generator = val_datagen.flow_from_dataframe(val_df, directory=INPUT_IMG_PATH, x_col='filename', y_col='latex',
                                                target_size=IMG_SIZE, batch_size=32, class_mode='categorical')
test_generator = test_datagen.flow_from_dataframe(test_df, directory=INPUT_IMG_PATH, x_col='filename', y_col='latex',
                                                  target_size=IMG_SIZE, batch_size=32, class_mode='categorical')

for data_batch, labels_batch in train_generator:
    print('train data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)

for data_batch, labels_batch in val_generator:
    print('val data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)

for data_batch, labels_batch in test_generator:
    print('test data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)

### MODEL DEFINITION
### VGG16 model
def our_model():
    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', input_shape=(224,224,3))) 
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu')) 
    model.add(MaxPooling2D(pool_size=(2,2), strides=2))

    filters_convs = [(128, 2), (256, 3), (512, 3), (512,3)] 
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
    model.compile(loss = keras.categorical_crossentropy, optimizer = opt, metrics = ['accuracy'])

    return model
