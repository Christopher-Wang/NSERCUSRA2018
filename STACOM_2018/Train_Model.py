#Import necessary files from local modules
from Data.DataGenerator import *
from Models import LossFunctions as loss
from Models import Ensemble_3D as Architecture
from keras.models import *
from keras.callbacks import ModelCheckpoint
import keras.losses
keras.losses.jaccard_loss = loss.jaccard_loss
import keras.metrics
keras.metrics.dice = loss.dice
import os
import numpy as np
import tensorflow as tf
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import numpy as np


block = sys.argv[1]
merge = sys.argv[2]
version = sys.argv[3]
message = block + " U-Net with " + merge +" skip connections"
model_version = "-".join([block, merge, version])
#HyperParameters
filters = int(sys.argv[4])

#Augmentation Parameters
preprocess = 'standardize'
augmentations = ['translate', 'rotate', 'flip', 'offset', 'noise','elastic_transform']

#Load the label data
dataFilePath = "Data/Dataset"
modelFilePath = "Models/" 
predictionFilePath = "Predictions/"
labels = np.load(dataFilePath + "__labels__.npz")
labels = dict(labels)['labels']
training_subset = labels[20:]
np.random.shuffle(training_subset)
training, testing, validation = training_subset[10:], labels[:20], training_subset[:10]

#Build the generators
trainingAugmentation = DataAugmenter(preprocess, augmentations)
testingAugmentation = DataAugmenter(preprocess, [])

trainingGenerator = DataGenerator(dataFilePath, training, trainingAugmentation)
validationGenerator = DataGenerator(dataFilePath, validation, trainingAugmentation)
testingGenerator = DataGenerator(dataFilePath, testing, testingAugmentation, shuffle = False)
predictionGenerator = DataGenerator(dataFilePath, testing, testingAugmentation, shuffle = False)


#Build the model architecture
model = Architecture.UNET(block, merge, filters=filters)
#model.load_weights(modelFilePath + model_version + '.hdf5')
model.compile(optimizer = "Adam", loss = loss.jaccard_loss, metrics=[loss.dice])
print(model_version)
print(message)
model.summary()

checkpoint = ModelCheckpoint(modelFilePath + model_version + '.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit_generator(trainingGenerator, 
					steps_per_epoch=70, 
					epochs = 70,
  			  validation_data = validationGenerator, 
					validation_steps = 20,
          callbacks=callbacks_list,
          verbose=2)

model.load_weights(modelFilePath + model_version + '.hdf5')
predictions = model.predict_generator(predictionGenerator, steps=20)

print("--Results--")
for i in range(20):
	_, mask = testingGenerator.__next__()
	dice = 2. * np.sum(predictions[i] * mask) / np.sum(predictions[i] + mask)
	print(dice)

np.save(predictionFilePath + model_version + ".npy", predictions)