import os
import numpy as np
import pandas as pd
import tensorflow as tf

# image readers/processors
import cv2
from PIL import Image
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

### GLOBAL VARIABLES

# INPUT IMAGE STANDARD SIZE
IMG_HEIGHT = 100
IMG_WIDTH = 275
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# absolute file paths
TRAIN_LST_PATH = '/Users/tuyet/Documents/S23/CSC 561/Final Project/Final/dumasnguyen_csc561/im2latex/data/im2latex_train.lst'
VAL_LST_PATH = '/Users/tuyet/Documents/S23/CSC 561/Final Project/Final/dumasnguyen_csc561/im2latex/data/im2latex_validate.lst'
TEST_LST_PATH = '/Users/tuyet/Documents/S23/CSC 561/Final Project/Final/dumasnguyen_csc561/im2latex/data/im2latex_test.lst'
FORMULAS_LST_PATH = '/Users/tuyet/Documents/S23/CSC 561/Final Project/Final/dumasnguyen_csc561/im2latex/data/im2latex_formulas.lst'
IMAGE_PATH = '/Users/tuyet/Documents/S23/CSC 561/Final Project/Final/dumasnguyen_csc561/im2latex/data/formula_images/'

def create_df_from_data_lst(lst_file, formulas_list):
    """
    Create a Pandas DataFrame from data lst file

    Parameters
    ----------
    lst_file: string
        full file path to lst file, to be read only

    formulas_list: list
        full list of latex formulas strings

    Returns
    ----------
    df: DataFrame
        A Pandas DataFrame containing all info read from lst
    """
    data = {}
    data['formulas_idx'] = []
    data['image'] = []
    data['render_type'] = []
    data['latex'] = []

    with open(lst_file, newline='\n') as file:
        for line in file:
            idx, img, rend = line.strip().split(' ')
            data['formulas_idx'].append(int(idx))
            data['image'].append(IMAGE_PATH+img+'.png')
            data['render_type'].append(rend)
            data['latex'] = formulas_list[int(idx)]

    df = pd.DataFrame.from_dict(data)
    return df


def create_df_from_formulas_lst(lst_file):
    """
    Create list of latex strings from formulas lst file

    Parameters
    ----------
    lst_file: string
        full file path to lst file, to be read only

    Returns
    ----------
    formulas: list
        list of latex formulas read fromlst
    """
    formulas = []

    with open(lst_file, newline='\n', encoding='latin-1') as file:
        formulas = [line.strip() for line in file]

    return formulas


def get_dataloaders():
    """
    Given all globals above are correct, wil return three Keras ImageDataGenerator objects

    Returns
    ----------
    train_dataloader: Keras ImageDataGenerator
        dataloader for pre-defined training data
    val_dataloader: Keras ImageDataGenerator
        dataloader for pre-defined validation data
    test_dataloader Keras ImageDataGenerator
        dataloader for pre-defined testing data
    """

    # Load all lst file data into a Pandas DataFrame
    formulas = create_df_from_formulas_lst(FORMULAS_LST_PATH)
    train_df = create_df_from_data_lst(TRAIN_LST_PATH, formulas)
    val_df = create_df_from_data_lst(VAL_LST_PATH, formulas)
    test_df = create_df_from_data_lst(TEST_LST_PATH, formulas)

    for df in [train_df, val_df, test_df]:
        print(df.head())

    # create dataloaders from dataframes
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_dataloader = train_datagen.flow_from_dataframe(train_df, directory=IMAGE_PATH, x_col='image', y_col='latex',
                                                        target_size=IMG_SIZE, batch_size=32)
    val_dataloader = val_datagen.flow_from_dataframe(val_df, directory=IMAGE_PATH,  x_col='image', y_col='latex',
                                                    target_size=IMG_SIZE, batch_size=32)
    test_dataloader = test_datagen.flow_from_dataframe(test_df, directory=IMAGE_PATH,  x_col='image', y_col='latex',
                                                    target_size=IMG_SIZE, batch_size=32)
    
    return train_dataloader, val_dataloader, test_dataloader
