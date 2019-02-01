"""
A script that converts all of the STACOM_2018 .nrrd files into numpy arrays and saves it as an npz.

The format of the .npz will be as follows:
When first loaded, an NpzFile object will be returned. It will have 3 keys, labels, raws, masks.
labels is the name of the sample (String)
raws is the raw MRI file as a 3D matrix (numpy array)
masks is the masked MRI file as a 3D matrix (numpy array)

Each key will return a np array containing all of the 100 training data's label, raw and masks
"""
import numpy as np
import SimpleITK as sitk
import xlrd

from matplotlib import pyplot as plt
from scipy import ndimage as sp

#Constant values from dataset
numSlides = 88
labelPosition = 0
labelSize = 20

rawFile = '/lgemri.nrrd'
maskFile = '/laendo.nrrd'

# TAKEN FROM THE SAMPLE CODE
# This function loads .nrrd files into a 3D matrix and outputs it
# 	the input is the specified file path to the .nrrd file
def load_nrrd(full_path_filename):
	data = sitk.ReadImage( full_path_filename )
	data = sitk.Cast( sitk.RescaleIntensity(data), sitk.sitkUInt8 )
	data = sitk.GetArrayFromImage(data)
	return(data)

#Get the labels
def getLabels(filepath):
	labelBook = xlrd.open_workbook(filepath)
	labels = []
	for sheet in labelBook.sheets():
		for row in range(1, sheet.nrows, numSlides):
			labels.append(sheet.row(row)[labelPosition].value[:labelSize])
	return labels

#Process the data
def preprocess(data):
	#Pad if necessary
	if data.shape[1] == 576:
		data = np.pad(data, ((0, 0),(32, 32),(32, 32)), 'constant')
	data = data[:,160:-160, 160:-160]
	data = data.reshape(*data.shape, 1)
	data = data.astype('float32')
	data = sp.zoom(data, (1, 0.5, 0.5, 1))
	#Standardize the image
	mean = np.mean(data, axis = (1, 2))
	std = np.std(data, axis = (1, 2))
	data -= np.reshape(mean, (88, 1, 1, 1))
	data /= np.reshape(std, (88, 1, 1, 1))
	return data

#Save the data in a single .npz file
def saveChunk(filename, sampleNumber = 100, z1 = 0, z2 = 88, x1 = 0, x2 = 640, y1 = 0, y2 = 640):
	labels = getLabels()
	raws = []
	masks = []

	#Get the array data
	for label in labels:
		#Load the data
		raw = load_nrrd(filepath + label + rawFile)[z1:z2][x1:x2][y1:y2]
		mask = load_nrrd(filepath + label + maskFile)[z1:z2][x1:x2][y1:y2]
		#Process the data
		raw = preprocess(raw)
		mask = preprocess(mask)
		#Append the values to e stored
		raws.append(raw)
		masks.append(mask)

	#Convert labels to arrays	
	labels = np.array(labels)
	raw = np.array(raws)
	masks = np.array(masks)

	#Save the values, subject to change. Right now saving a smaller subset of data to use for iteration
	np.savez(savepath + filename +'.npz', labels = labels[:sampleNumber], raws = raws[:sampleNumber], masks = masks[:sampleNumber])

#Save individual files
def saveIndividual(filename, sampleNumber = 100, z1 = 0, z2 = 88, x1 = 0, x2 = 640, y1 = 0, y2 = 640):
	labels = getLabels()
	#Get the array data
	for label in labels:
		#For now take slices 44-74 to reduce size
		raw = load_nrrd(filepath + label + rawFile)[z1:z2][x1:x2][y1:y2]
		mask = load_nrrd(filepath + label + maskFile)[z1:z2][x1:x2][y1:y2]
		raw = preprocess(raw)
		mask = preprocess(mask)
		np.savez(savepath + label + '.npz', raw = raw, mask = mask)

#Function that loads a full .npz files and unloads the data
def loadData(filename):
	data = np.load(filename)
	labels, raws, masks = dict(data).values()
	#Reshape the data into a form keras can understand (timestep, x, y, channels)
	return labels, raws, masks



