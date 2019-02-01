"""
This is the script to be run on SHARCNET to augment the STACOM 2018 left atrium MRI dataset.
Code learned https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
"""
import numpy as np
import random 
from scipy import ndimage as sp
from inspect import signature
import cv2


"""
A class that Defines the augmentation parameters of the Data Generator
"""
class DataAugmenter(object):
	def __init__(self, preprocess, augmentations=[], 
		augment_chance=1, 
		shift_range = (-10, 11), 
		rotation_range = (-45, 46), 
		offset_sigma=0.01, 
		noise_sigma=0.0005,
		transform_sigma = 20,
		transform_alpha = 1):
		preprocess_functions = {'normalize': self.normalize, 'normalize2D': self.normalize2D,'standardize': self.standardize,'standardize2D': self.standardize}
		augmentation_functions = {'translate': self.translate, 'rotate': self.rotate, 'flip': self.flip, 'offset': self.offset, 'noise': self.noise, 'elastic_transform':self.elastic_transform}
		self.preprocess = preprocess_functions[preprocess]
		self.augmentations = [augmentation_functions[function] for function in augmentations]
		self.augment_chance = augment_chance
		self.shift_range = shift_range
		self.rotation_range = rotation_range
		self.offset_sigma = offset_sigma
		self.noise_sigma = noise_sigma
		self.transform_sigma = transform_sigma
		self.transform_alpha = transform_alpha

	def normalize(self, data):
		data = (data - np.amin(data))/ (np.amax(data) - np.amin(data))
		return data

	def normalize2D(self, data):
		amin = np.reshape(np.amin(data, axis = (2, 3)), (data.dim[0], 1, 1, data.dim[-1])) 
		amax = np.reshape(np.amax(data, axis = (2, 3)), (data.dim[0], 1, 1, data.dim[-1])) 
		data = (data - amin) / (amax - amin)
		return data

	def standardize(self, data):
		mean = np.mean(data)
		std = np.std(data)
		data -= mean
		data /= std
		return data

	def standardize2D(self, data):
		mean = np.mean(data, axis = (2, 3))
		std = np.std(data, axis = (2, 3))
		data -= np.reshape(mean, (data.dim[0], 1, 1, data.dim[-1])) 
		data /= np.reshape(std, (data.dim[0], 1, 1, data.dim[-1])) 
		return data

	#Augmentation Functions to both the raw and the mask
	def translate(self, raw, mask):
		if np.random.random(1) < self.augment_chance:
			width_shift = random.randint(*self.shift_range)
			height_shift = random.randint(*self.shift_range)
			raw = np.roll(raw, (width_shift, height_shift), axis = (2, 1))
			mask = np.roll(mask, (width_shift, height_shift), axis = (2, 1))
		return raw, mask

	def rotate(self, raw, mask):
		if np.random.random(1) < self.augment_chance:
			rotation = rotation = random.randint(*self.rotation_range)
			raw = sp.rotate(raw, rotation, (1, 2), reshape = False)
			mask = sp.rotate(mask, rotation, (1, 2), reshape = False)
		return raw, mask

	def flip(self, raw, mask):
		if np.random.random(1) < self.augment_chance:
			axis = np.random.randint(2, 4)
			raw = np.flip(raw, axis)
			mask = np.flip(mask, axis)
		return raw, mask
    
	def elastic_transform(self, raw, mask):
		data = np.concatenate((raw, mask), axis = -1)
		out = np.zeros((data.ndim, *data.shape))
		# Generate a Gaussian filter, leaving channel dimensions zeroes
		print(self.transform_alpha)
		for i, alpha in enumerate(self.transform_alpha):
			array = (np.random.rand(*data.shape) * 2 - 1)
			out[i] = sp.filters.gaussian_filter(array, self.transform_sigma[i], mode="constant", cval=0) * alpha
			shapes = list(map(lambda x: slice(0, x, None), data.shape))
			grid = np.broadcast_arrays(*np.ogrid[shapes])
			indices = list(map((lambda x: np.reshape(x, (-1, 1))), grid + np.array(out)))
			data = sp.interpolation.map_coordinates(data, indices, order=0, mode='reflect').reshape(data.shape)
		raw, mask = data[:,:,:,0], data[:,:,:,1]
		mask = (mask > 0.5).astype('float32')
		return np.reshape(raw, (*raw.shape, 1)), np.reshape(mask, (*mask.shape, 1))

	#Augmentaion Functions to preform only to the raw
	def offset(self, raw):
		if np.random.random(1) < self.augment_chance:
			offset = np.random.normal(0, self.offset_sigma, ([1] * (raw.ndim - 1) + [raw.shape[-1]]))
			raw += offset
		return raw

	def noise(self, raw):
		if np.random.random(1) < self.augment_chance:
			raw += np.random.normal(0, self.noise_sigma, raw.shape)
		return raw


"""
An iterator class that reads data from files and outputs the files in the appropriate batch size. This is to be used
dynamically when training to save space.
"""
class DataGenerator(object):
	def __init__(self, path, labels, DataAugmenter=None, batch_size=1, dim = (88, 160, 160, 1), shuffle = True):
		self.path = path
		self.labels = labels
		self.DataAugmenter = DataAugmenter
		self.batch_size = batch_size
		self.dim = dim
		self.indices = np.arange(len(self.labels))
		self.index = 0
		self.shuffle = shuffle

	#Shuffle when starting a new Epoch and increment the index
	def _increment(self):
		if self.shuffle and self.index == 0:
			np.random.shuffle(self.indices)
		self.index  = (self.index + 1) % self.__len__()

	#Load the data from an example
	def _data_load(self, label):
		data = dict(np.load(self.path + label + '.npz'))
		raw = data['raw']
		mask = data['mask']
		return raw, mask

	#Produce batches of data
	def _data_generation(self, labels):
		#Create the raws and masks arrays
		raws = np.empty((self.batch_size, *self.dim))
		masks = np.empty((self.batch_size, *self.dim))
		#Get all the values from 
		for i, label in enumerate(labels):
			raw, mask = self._data_load(label)
			if self.DataAugmenter.augmentations:
			 	raw, mask = self._data_augmentation(raw, mask)
			raw = self.DataAugmenter.preprocess(raw)
			raws[i] = raw
			raws[i], masks[i] = raw, mask
		return raws, masks

	def _data_augmentation(self, raw, mask):
		np.random.shuffle(self.DataAugmenter.augmentations)
		for augmentation in self.DataAugmenter.augmentations:
			if len(signature(augmentation).parameters) > 1:
				raw, mask =  augmentation(raw, mask)
			else: 
				raw = augmentation(raw)
		return raw, mask

	#Get the number of batches per epoch
	def __len__(self):
		#The floor of the total length/batchsize
		return int(np.floor(len(self.labels) / self.batch_size)) 

	#Generate the batches corresponding to a given example
	def __next__(self):
		#Slice the appropriate indices
		indices = self.indices[self.index * self.batch_size:(self.index + 1) * self.batch_size]
		self._increment()
		#Create the list of labels from the indices subsection
		labels = [self.labels[i] for i in indices]
		#Get the data
		# raws, masks = self._data_generation(labels)
		raws = self._data_generation(labels)
		return raws#, masks