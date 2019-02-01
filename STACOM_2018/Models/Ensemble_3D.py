"""
A Convolutional LSTM implementation of UNET architecture
"""
#Import necessary libraries
import numpy as np
import tensorflow as tf

from keras.models import Model
from keras.layers import *

def DenseBlock(input, filters, i):
	input = Input(shape=input.shape.as_list()[1:])
	dense0 = Conv3D(filters, (3, 3, 3), activation = 'relu', padding = 'same')(input)
	dense0 = BatchNormalization()(dense0)

	dense1 = Conv3D(filters, (3, 3, 3), activation = 'relu', padding = 'same')(dense0)
	dense1 = BatchNormalization()(dense1)
	dense1 = concatenate([dense0, dense1], axis = -1)

	dense2 = Conv3D(filters, (3, 3, 3), activation = 'relu', padding = 'same')(dense1)
	dense2 = BatchNormalization()(dense2)
	dense2 = concatenate([dense1, dense2], axis = -1)

	dense3 = Conv3D(filters, (3, 3, 3), activation = 'relu', padding = 'same')(dense2)
	dense3 = BatchNormalization()(dense3)
	dense3 = concatenate([dense2, dense3], axis = -1)

	output = Conv3D(filters, (1, 1, 1), activation = 'relu', padding = 'same')(dense3)
	model = Model(inputs = [input], outputs = [output], name = 'Dense_' + str(i))
	return model

def ResidualBlock(input, filters, i, transition = True):
	input = Input(shape=input.shape.as_list()[1:])
	residual = Conv3D(filters, (3, 3, 3), activation = 'relu', padding = 'same')(input)
	residual = BatchNormalization()(residual)

	residual = Conv3D(filters, (3, 3, 3), padding = 'same')(residual)
	residual = BatchNormalization()(residual)

	if transition:
		output = Conv3D(filters, (1, 1, 1), activation = 'relu', padding = 'same')(input)
		output = add([residual, output])
	else:
		output= add([residual, input])

	output = Activation('relu')(output)
	model = Model(inputs = [input], outputs = [output], name = 'Residual_' + str(i))
	return model

def SqueezeExcitationModel(shape, filters):
	input = Input(shape=shape)
	se = GlobalAveragePooling3D()(input)
	se = Dense(filters, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
	se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
	output = multiply([input, se])
	model = Model(inputs = [input], outputs = [output])
	return model

def SqueezeExcitatioBlock(input, filters, i):
	input = Input(shape=input.shape.as_list()[1:])
	conv = Conv3D(filters, (3, 3, 3), activation = 'relu', padding = 'same')(input)
	se = SqueezeExcitationModel(conv.shape.as_list()[1:], filters)(conv)
	se = BatchNormalization()(se)
	conv = Conv3D(filters, (3, 3, 3), activation = 'relu', padding = 'same')(se)
	output = SqueezeExcitationModel(conv.shape.as_list()[1:], filters)(conv)
	model = Model(inputs = [input], outputs = [output], name = 'SqueezeExcitation_' + str(i))
	return model

def ConvolutionBlock(input, filters, i):
	input = Input(shape=input.shape.as_list()[1:])
	conv = Conv3D(filters, (3, 3, 3), activation = 'relu', padding = 'same')(input)
	conv = BatchNormalization()(conv)
	output = Conv3D(filters, (3, 3, 3), activation = 'relu', padding = 'same')(conv)
	model = Model(inputs = [input], outputs = [output], name = 'Convolution_' + str(i))
	return model

def InceptionBlock(input, filters, i):
	input = Input(shape=input.shape.as_list()[1:])
	inception0 = Conv3D(int(filters/4), (1, 1, 1), activation = 'relu', padding = 'same')(input)
	inception0 = BatchNormalization()(inception0)
	inception0 = Conv3D(int(filters/4), (1, 1, 1), activation = 'relu', padding = 'same')(inception0)

	inception1 = Conv3D(int(filters/4), (1, 1, 1), activation = 'relu', padding = 'same')(input)
	inception1 = BatchNormalization()(inception1)
	inception1 = Conv3D(int(filters/2), (3, 3, 3), activation = 'relu', padding = 'same')(inception1)

	inception2 = Conv3D(int(filters/4), (1, 1, 1), activation = 'relu', padding = 'same')(input)
	inception2 = BatchNormalization()(inception2)
	inception2 = Conv3D(int(filters/8), (3, 3, 3), activation = 'relu', padding = 'same')(inception2)
	inception2 = BatchNormalization()(inception2)
	inception2 = Conv3D(int(filters/8), (3, 3, 3), activation = 'relu', padding = 'same')(inception2)

	inception3 = MaxPooling3D(pool_size = (3, 3, 3), padding = 'same', strides = (1, 1, 1))(input)
	inception3 = Conv3D(int(filters/8), (1, 1, 1), activation = 'relu', padding = 'same')(inception3)

	output = concatenate([inception0, inception1, inception2, inception3], axis=-1)
	model = Model(inputs = [input], outputs = [output], name = 'Inception_' + str(i))
	return model


def addBlock(input0, input1, filters, i):
	input0 = Input(shape=input0.shape.as_list()[1:])
	input1 = Input(shape=input1.shape.as_list()[1:])

	reshape = Conv3D(filters, (1, 1, 1), activation = 'relu', padding = 'same')(input1)
	output = add([input0, reshape])

	model = Model(inputs = [input0, input1], outputs = [output], name = 'add_' + str(i))
	return model

def concatBlock(input0, input1, filters, i):
	input0 = Input(shape=input0.shape.as_list()[1:])
	input1 = Input(shape=input1.shape.as_list()[1:])

	reshape = Conv3D(filters, (1, 1, 1), activation = 'relu', padding = 'same')(input1)
	output = concatenate([input0, reshape], axis=-1)

	model = Model(inputs = [input0, input1], outputs = [output], name = 'concat_' + str(i))
	return model

def attentionBlock(input0, input1, filters, i):
	input0 = Input(shape=input0.shape.as_list()[1:])
	input1 = Input(shape=input1.shape.as_list()[1:])

	layer1 = Conv3D(filters, (1, 1, 1), padding = 'same')(input1)
	layer2 = BatchNormalization()(input0)
	layer2 = Conv3D(filters, (1, 1, 1), padding = 'same')(layer2)
	gate = add([layer2, layer1])
	gate = Activation('relu')(gate)
	gate = BatchNormalization()(gate)
	gate = Conv3D(filters, (1, 1, 1), activation = 'sigmoid', padding = 'same')(gate)
	output = multiply([input0, gate])
	output = concatenate([output, input1])

	model = Model(inputs = [input0, input1], outputs = [output], name = 'attention_' + str(i))
	return model


def UNET(block_type, merge_type, dim = (88, 160, 160, 1), filters = 16):
	#General parameter choice
	block = {'Dense': DenseBlock, 'Residual': ResidualBlock,  'Inception': InceptionBlock, 'SqueezeExcitation': SqueezeExcitatioBlock, 'Convolution': ConvolutionBlock}
	block = block[block_type]
	merge = {'add': addBlock, 'concat': concatBlock, 'attention': attentionBlock}
	merge = merge[merge_type]

	input = Input(shape = dim)
	layer0 = block(input, filters, 0)(input)
	pool = MaxPooling3D()(layer0)

	layer1 = BatchNormalization()(pool)
	layer1 = block(layer1, filters * 2, 1)(layer1)
	pool = MaxPooling3D()(layer1)

	layer2 = BatchNormalization()(pool)
	layer2 = block(layer2, filters * 4, 2)(layer2)
	pool = MaxPooling3D()(layer2)

	layer3 = BatchNormalization()(pool)
	layer3 = block(layer3, filters * 8, 3)(layer3)
	upsample = UpSampling3D()(layer3)

	layer4 = BatchNormalization()(upsample)
	layer4 = merge(layer2, layer4, filters * 4, 0)([layer2, layer4])
	layer4 = BatchNormalization()(layer4)
	layer4 = block(layer4, filters* 4, 4)(layer4)
	upsample = UpSampling3D()(layer4)

	layer5 = BatchNormalization()(upsample)
	layer5 = merge(layer1, layer5, filters * 2, 1)([layer1, layer5])
	layer5 = BatchNormalization()(layer5)
	layer5 = block(layer5, filters * 2, 5)(layer5)
	upsample = UpSampling3D()(layer5)

	layer6 = BatchNormalization()(upsample)
	layer6 = merge(layer0, layer6, filters, 2)([layer0, layer6])
	layer6 = BatchNormalization()(layer6)
	layer6 = block(layer6, filters, 6)(layer6)
	layer6 = BatchNormalization()(layer6) 
#	layer6  = ConvLSTM2D(8, (3, 3), return_sequences=True, padding = 'same')(layer6) 
	output = ConvLSTM2D(1, (3, 3), activation = 'sigmoid', return_sequences=True, padding = 'same')(layer6) 
#	output = Conv3D(1, (1, 1, 1), activation = 'sigmoid', padding = 'same')(layer6)

	#Finalize the model
	model = Model(inputs=[input], outputs=[output])
	return model

def CNET(block_type, merge_type, dim = (88, 160, 160, 1), filters = 16):
	#General parameter choice
	block = {'Dense': DenseBlock, 'Residual': ResidualBlock,  'Inception': InceptionBlock, "SqueezeExcitation": SqueezeExcitatioBlock, "Convolution": ConvolutionBlock}
	block = block[block_type]
	merge = {'add': addBlock, 'concat': concatBlock, 'attention': attentionBlock}
	merge = merge[merge_type]

	input = Input(shape = dim)
	layer0 = block(input, filters, 0)(input)
	pool = MaxPooling3D()(layer0)

	layer1 = BatchNormalization()(pool)
	layer1 = block(layer1, filters * 2, 1)(layer1)
	pool = MaxPooling3D()(layer1)

	layer2 = BatchNormalization()(pool)
	layer2 = block(layer2, filters * 4, 3)(layer2)
	layer2 = BatchNormalization()(pool)
	layer2 = block(layer2, filters * 4, 4)(layer2)
	upsample = UpSampling3D()(layer2)

	layer3 = BatchNormalization()(upsample)
	layer3 = merge(layer1, layer3, filters * 2, 0)([layer1, layer3])
	layer3 = BatchNormalization()(layer3)
	layer3 = block(layer3, filters * 2, 5)(layer3)
	upsample = UpSampling3D()(layer3)

	layer4 = BatchNormalization()(upsample)
	layer4 = merge(layer0, layer4, filters, 1)([layer0, layer4])
	layer4 = BatchNormalization()(layer4)
	layer4 = block(layer4, filters, 6)(layer4) 
#	layer4  = ConvLSTM2D(8, (3, 3), return_sequences=True, padding = 'same')(layer4) 
#	layer4  = ConvLSTM2D(8, (3, 3), return_sequences=True, padding = 'same')(layer4) 
	output = Conv3D(1, (1, 1, 1), activation = 'sigmoid', padding = 'same')(layer4)

	#Finalize the model
	model = Model(inputs=[input], outputs=[output])
	return model
