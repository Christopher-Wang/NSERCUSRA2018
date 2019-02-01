#Tutorial taken from https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
#Import necessary files
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Convolution1D, MaxPooling1D
#Embedding is a way to map words into real value vectors of length 32. 
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
np.random.seed(7)

#Number of words being modeled 
numWords = 5000
#Number of words in each review
reviewWordCount = 500
#Size of the word embeddings vector space
wordSize = 32
#Load the data into tuples
(xTrain, yTrain), (xTest, yTest) = imdb.load_data(num_words=numWords)

#Preprocessing
#Pad the smaller reviews with empty words
xTrain = sequence.pad_sequences(xTrain, maxlen = reviewWordCount)
xTest = sequence.pad_sequences(xTest, maxlen = reviewWordCount)

#Build the model
model = Sequential([
	#Embed the words in a wordSize dimensional space
	Embedding(numWords, wordSize, input_length = reviewWordCount),
	Dropout(0.2),
	#Add a 1D convolution layer, give the output the same size as the input vector size 
	Convolution1D(32, 3, padding = 'same',activation = 'relu'),
	#Recurrent layer, also has a specialized recurrent drop out
	LSTM(100, recurrent_dropout = 0.25),
	#Output layer, binary output (review good or bad?)
	Dense(1, activation = 'sigmoid')
	])

#Compile the model
model.compile(
	loss = 'binary_crossentropy',
	optimizer = 'adam',
	metrics = ['accuracy']
	)

#Train the model
model.fit(xTrain, yTrain, epochs = 3, batch_size = 50)

#Test the model
scores = model.evaluate(xTest, yTest, verbose = 1)