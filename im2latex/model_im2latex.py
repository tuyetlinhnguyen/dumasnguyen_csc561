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
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

# custom modules
from preprocess_data import get_dataloaders
from encoder import our_encoder
from global_ import IMAGE_SIZE, BATCH_SIZE, NUM_EPOCHS, LR

import wandb
# start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="csc561_model_test",
    
#     # track hyperparameters and run metadata
#     config={
#     "learning_rate": LR,
#     "architecture": "CNN-RNN",
#     "dataset": "im2latex-100k",
#     "epochs": NUM_EPOCHS,
#     }
# )

train_dataloader, val_dataloader, test_dataloader = get_dataloaders()

# model = our_encoder()

# # Train model on dataset
# training_history = model.fit(x=train_dataloader,
#                     validation_data=val_dataloader,
#                     batch_size=BATCH_SIZE,
#                     epochs=NUM_EPOCHS,
#                     use_multiprocessing=True,
#                     workers=4
#                    )

# print(training_history.history)

# test_loss = model.evaluate(x=test_dataloader, 
#                            batch_size=BATCH_SIZE, 
#                            use_multiprocessing=False,
#                            workers=4
#                            )

# print(test_loss)
