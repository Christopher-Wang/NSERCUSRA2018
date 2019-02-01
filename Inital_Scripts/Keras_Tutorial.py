#Tutorial taken from https://elitedatascience.com/keras-tutorial-deep-learning-in-python
#These are important libraries
import numpy as numpy
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.utils import np_utils
#The loaded data
from keras.datasets import mnist



#Load the date into tuples
(xTrain, yTrain), (xTest, yTest) = mnist.load_data()

#Preprocess the data
#Reshape the data into a form keras can understand (x, y, z)
xTrain = xTrain.reshape(xTrain.shape[0], 28, 28, 1)
xTest = xTest.reshape(xTest.shape[0], 28, 28, 1)
#Reshape the labes into a form that makes sense (Not just the label but an array containing the labels)
#For example, instead of 5 it is [0,0,0,0,0,1,0,0,0,0]
yTrain = np_utils.to_categorical(yTrain, 10)
yTest = np_utils.to_categorical(yTest, 10)
#Convert the data to float
#I think this is because we are using 8bit greyscale and therefore the values are int
xTrain = xTrain.astype('float32')
xTest = xTest.astype('float32')
#Normalize the data (assuming 8-bit greyscale)
xTrain /= 255
xTest /= 255

#Build the model architecture
model = Sequential([
	#(Number of filters, filter size, activation, input (only for first layer)
	Convolution2D(24, (3, 3), activation = 'relu', input_shape = (28,28, 1)),
	Convolution2D(24, (5, 5), activation = 'relu'),
	Flatten(),
	Dropout(0.25),
	Dense(128, activation = 'relu'),
	Dropout(0.25),
	Dense(124, activation = 'relu'),
	Dropout(0.25),
	Dense(10, activation  = 'softmax')
	])

#Compile the model (loss function, optimization strategy, metric)
model.compile(
	loss = 'categorical_crossentropy', 
	optimizer = 'SGD', 
	metrics = ['accuracy']
	)

#Train the model
print(xTrain[1])
model.fit(xTrain, yTrain, batch_size = 32, epochs= 3, verbose = 1)
#Test the model
score = model.evaluate(xTest, yTest, verbose = 0)
print(score)