#!/usr/bin/python3

from models_and_puzzle.models.sudokunet import SudokuNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m","--model" , required=True , help="path to output model after training")
args = vars(ap.parse_args())

# Learning rate 
ini_lr = 1e-3
EPOCHS = 10
bs = 128

# MNIST dataset
print("[INFO] accessing MNIST.......")
((traindata , trainlabels) , (testdata , testlabels)) = mnist.load_data()

# Changing shape acc. to keras , or adding channel dimension
traindata = traindata.reshape((traindata.shape[0] , 28,28,1))
testdata = testdata.reshape((testdata.shape[0] , 28,28,1))

# scaling data
traindata = traindata.astype("float32") / 255.0
testdata = testdata.astype("float32") / 255.0

# Converting Labels from integers to vectors
le = LabelBinarizer()
trainlabels = le.fit_transform(trainlabels)
testlabels = le.fit_transform(testlabels)

# initialize the optimizer and model
print("[INFO] compiling model......")
opt = Adam(lr = ini_lr)
model = SudokuNet.build(width=28,height=28,depth=1,classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt , metrics=["accuracy"])

# train the network
print("[INFO] training network....")
mo = model.fit(traindata,trainlabels,validation_data=(testdata,testlabels) , batch_size=bs , epochs=EPOCHS , verbose=1)


# Evaluating the network
print("[INFO] Evaluating the n/w....")
pred = model.predict(testdata)
print(classification_report(testlabels.argmax(axis=1) ,pred.argmax(axis=1) , target_names = [str(x) for x in le.classes_] ))

# saving the model
print("[INFO] serializing digit model / saving ......")
model.save(args["model"] , save_format="h5")

