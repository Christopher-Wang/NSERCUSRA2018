from keras import backend as k
import tensorflow as tf
 
def union(mask, predicted):
	mask = k.flatten(k.abs(mask))
	predicted = k.flatten(k.abs(predicted))
	return k.sum(mask + predicted)

def inter(mask, predicted):
	mask = k.flatten(k.abs(mask))
	predicted = k.flatten(k.abs(predicted))
	return k.sum(mask * predicted)
 
def cardin(mask, predicted):
	mask = k.flatten(k.abs(mask))
	predicted = k.flatten(k.abs(predicted))
	return k.sum(mask) + k.sum(predicted)

def dice(mask, predicted):
	return 2. * inter(mask, predicted) / cardin(mask, predicted)

def dice_loss(mask, predicted):
	dice1 = dice(mask, predicted)
	return 1. - dice1

def jaccard(mask, predicted):
	return inter(mask, predicted) / (union(mask, predicted) - inter(mask, predicted))

def jaccard_loss(mask, predicted):
	return 1. - jaccard(mask, predicted)