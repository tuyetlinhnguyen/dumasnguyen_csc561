import os
import numpy as np
import pandas as pd
import tensorflow as tf

# image readers/processors
from PIL import Image
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

from global_ import IMAGE_SIZE, TRAIN_LST_PATH, VAL_LST_PATH, TEST_LST_PATH, FORMULAS_LST_PATH, IMAGE_PATH, BATCH_SIZE


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
            data['image'].append(IMAGE_PATH+img+'.jpg')
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

    # for df in [train_df, val_df, test_df]:
    #     print(df.head())
    # print(train_df['latex'].head(10))

    # create dataloaders from dataframes
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_dataloader = train_datagen.flow_from_dataframe(train_df, directory=IMAGE_PATH, x_col='image', y_col='latex',
                                                        target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, color_mode='grayscale')
    val_dataloader = val_datagen.flow_from_dataframe(val_df, directory=IMAGE_PATH,  x_col='image', y_col='latex',
                                                    target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, color_mode='grayscale')
    test_dataloader = test_datagen.flow_from_dataframe(test_df, directory=IMAGE_PATH,  x_col='image', y_col='latex',
                                                    target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, color_mode='grayscale')
    
    return train_dataloader, val_dataloader, test_dataloader
