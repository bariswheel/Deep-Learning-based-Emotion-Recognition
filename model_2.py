'''
model_2.py defines a more complex neural network architecture
compared to model_1.py that combines both Convolutional Neural Networks (CNN) and 
Long Short-Term Memory (LSTM) layers using TensorFlow and Keras.

Purpose:
The purpose of model_2.py is to provide a hybrid model that leverages the strengths
of both CNNs and LSTMs. The CNN layers are used for feature extraction from images,
while the LSTM layers are used for processing temporal sequences of data. This model
is suitable for tasks that involve both spatial and temporal dependencies,
such as video classification or time-series analysis of image data.

Here’s how the binary classification fits in:

Valence: Classified into positive and negative.
Arousal: Classified into high and low.
Dominance: Classified into high and low.
Liking: Classified into like and unlike.

valence (how pleasant an emotion is) 
arousal (how activated or energized one feels). 
High dominance is associated with feelings of empowerment and control
Low dominance is associated with feelings of submission and powerlessness. 

'''

# import the necessary packages
from tensorflow.keras.models import Sequential # type:ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Input, Reshape, LSTM # type:ignore
from tensorflow.keras import backend as K # type:ignore

class networkArchFonc:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # Add an Input layer
        model.add(Input(shape=inputShape))

        model.add(Conv2D(filters=32, kernel_size=5, padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=2, strides=2, padding="same"))
        model.add(Conv2D(filters=64, kernel_size=5, padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=2, strides=2, padding="same"))
        model.add(Conv2D(filters=128, kernel_size=5, padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=2, strides=2, padding="same"))
        model.add(Flatten())
        model.add(Reshape((128, 4)))
        model.add(LSTM(128, return_sequences=True))
        model.add(LSTM(128, return_sequences=False))  # Only the last LSTM layer should have return_sequences=False
        model.add(Dense(100, activation="relu"))
        model.add(Dense(1, activation='sigmoid'))  # Change this line to have one unit with 'sigmoid' activation

        # return the constructed network architecture
        return model
