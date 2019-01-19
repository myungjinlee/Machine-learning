#!/usr/bin/env python3i

import numpy as np
import cv2
import matplotlib.pyplot as plt

data_i = 'downgesture_train.list'
data_o = 'downgesture_test.list'
lr = open(data_i,"r")
lr2 = open(data_o,"r")

files_i = lr.read().splitlines()
files_o = lr2.read().splitlines()
pixels =[]
Y=list()
Y_o = list()
for f in files_i:
    fil = cv2.imread(f,0)
    fil.reshape(1,960)
    pixels.append(fil)
    if 'down' in f:
        Y.append(1)
    else:
        Y.append(0)
ni=len(files_i)
Y = np.array(Y).reshape(184,1)
## train images pixels and target Y
pixels = np.array(pixels)
pixels = (pixels.T)/255
pixels = pixels.T.reshape(184,960)

## test images in pixels_o . Y_o is the corresponding test target 
pixels_o =[]
for f in files_o:
    fil2 = cv2.imread(f,0)
    fil2.reshape(1,960)
    pixels_o.append(fil2)
    if 'down' in f:
        Y_o.append(1)
    else:
        Y_o.append(0)
no=len(files_o)
Y_o = np.array(Y_o).reshape(83,1)
pixels_o = np.array(pixels_o)
pixels_o = (pixels_o.T)/255
pixels_o = pixels_o.T.reshape(83,960)

########################
# Create your first MLP in Keras

from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
#numpy.random.seed(7)
# load pima indians dataset
##dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
#X = dataset[:,0:8]
#Y = dataset[:,8]
# create model
model = Sequential()
model.add(Dense(100, input_dim=960, activation='sigmoid'))

model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(pixels, Y, epochs=150, batch_size=10)
# evaluate the model
scores = model.evaluate(pixels, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# calculate predictions
predictions = model.predict(pixels)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)

###########################################################
#########TEST#####################

# Fit the model
model.fit(pixels_o, Y_o, epochs=150, batch_size=10)
# evaluate the model
scores = model.evaluate(pixels_o, Y_o)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# calculate predictions
predictions = model.predict(pixels_o)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)

