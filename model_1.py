'''
model_1.py is a Python module that defines a simple Convolutional Neural Network (CNN) 
architecture using TensorFlow and Keras.

Purpose:
The purpose of model_1.py is to provide a straightforward CNN model that can be used for 
image classification tasks. This model includes basic layers such as convolutional layers,
activation layers, pooling layers, and fully connected layers to extract features from 
images and perform classification.

'''

# import the necessary packages

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tensorflow.keras import backend as K


class networkArchFonc:
    # Method below belongs to the class rather than instance of the class
    # Method's functionality is independent of any instance-specific data
    # As this is a utility method that performs a task in isolation. This
    # is the true for the other model 

    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(16, (2, 2), padding="same",
                         input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL layers
        # RELU: Rectified Linear Unit
        model.add(Conv2D(32, (2, 2), padding="same"))  # kernelere göre conv yerni bir matris oluşturma
        model.add(Activation("relu"))  # reulu: negatif değerleri çevirme relu sıfıra elu e üzeri
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())  # düzleştirme ?
        model.add(Dense(500))  # fully connected
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        #	print(model.summary())
        # return the constructed network architecture
        return model
