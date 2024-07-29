'''
model_2.py is a Python module that defines a more complex neural network architecture
compare to model_1.py that combines both Convolutional Neural Networks (CNN) and 
Long Short-Term Memory (LSTM) layers using TensorFlow and Keras.

Purpose:
The purpose of model_2.py is to provide a hybrid model that leverages the strengths
of both CNNs and LSTMs. The CNN layers are used for feature extraction from images,
while the LSTM layers are used for processing temporal sequences of data. This model
is suitable for tasks that involve both spatial and temporal dependencies,
such as video classification or time-series analysis of image data.

'''

# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Input, Conv1D, BatchNormalization, MaxPooling1D, LSTM, Reshape, RepeatVector, TimeDistributed, Dropout
from tensorflow.keras import backend as K

import argparse
from math import sqrt
from numpy import split, array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
import math
import datetime


class networkArchFonc:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        model.add(Conv2D(filters=32,
                         kernel_size=5,
                         padding="same",
                         activation="relu",
                         input_shape=inputShape))
        model.add(MaxPooling2D(pool_size=2, strides=2, padding="same"))
        model.add(Conv2D(filters=64, kernel_size=5, padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=2, strides=2, padding="same"))
        model.add(Conv2D(filters=128, kernel_size=5, padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=2, strides=2, padding="same"))
        model.summary()
        model.add(Reshape((128, 3), name='predictions'))
        model.add(LSTM(128, return_sequences=True))
        model.add(LSTM(128, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(100, activation="relu"))

        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        # return the constructed network architecture
        return model
