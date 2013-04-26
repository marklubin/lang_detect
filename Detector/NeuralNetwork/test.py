"""
test.py
NeuralNetwork tests

Mark Lubin
"""

import NeuralNetwork as NN
from ..configuration import *

def prediction_test(datafile,weightsfile):
	data = loadmat(datafile)
	X = data['X']
	y = data['y'] - 1
	y = np.array([i[0] for i in y])
	data = loadmat(weightsfile)
	theta = [data['Theta1'],data['Theta2']]
	N = NN.NeuralNetwork(theta=theta)
	print "Test data accurate to %f" % N.get_accuracy(X,y)
	C,G = N.cost_function(X,y,N.roll(theta))
	print "Cost a test Theta %f (should be near .3844)" % C

def training_test(fname):
	data = loadmat(fname)
	X = data['X']
	y = data['y'] - 1
	y = np.array([i[0] for i in y])
	layers = [400,25,10]
	N = NN.NeuralNetwork(layer_sizes=layers)
	N.train(X,y)
	print "Accurate to %f (should be > .93)" % N.get_accuracy(X,y)