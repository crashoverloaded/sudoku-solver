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
    # depth - Channels of MNIST digit images (28 px)
    # classes - the number of digits 0-9 (28 px)
        # initializing the model
        model = Sequential()
        inputshape = (height , width , depth)
        
