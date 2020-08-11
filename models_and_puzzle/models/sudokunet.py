#!/usr/bin/python3

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout


class SudokuNet:
    @staticmethod
    def build(width , height , depth , classes):
    # Width - width of an MNIST digit (28 px)
    # Height - Height of an MNIST digit (28 px)
    # depth - Channels of MNIST digit images (1 grayscale channel)
    # classes - the number of digits 0-9 (10 digits)
        # initializing the model
        model = Sequential()
        inputshape = (height , width , depth)
        
        # First set of layers
        model.add(Conv2D(32,(5,5) , padding='same',input_shape=inputshape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size =(2,2)))

        # Second set of layers
        model.add(Conv2D(32,(3,3) , padding='same'))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size =(2,2)))

        # First Set of FC layers
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        # Second set of FC
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        # Softmax
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model
