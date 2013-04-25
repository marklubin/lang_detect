"""
NeuralNetwork.py
MLP Neural Network with support for any number of hidden layers

Mark Lubin
"""


import numpy as np
from ..configuration import *
from math import isnan
from normalizer import normalize
from scipy.io import loadmat,savemat
from scipy.optimize import fmin_tnc as minimizer


"""
default activation function and gradient
"""
def sigmoid(X):
	return 1.0 / (1.0 + np.exp(-1. * X))

def sigmoidgradient(z):
	s = sigmoid(z)
	return s * (np.ones(z.shape) - s)

"""
tests
"""

def prediction_test(datafile,weightsfile):
	data = loadmat(datafile)
	X = data['X']
	y = data['y'] - 1
	y = np.array([i[0] for i in y])
	data = loadmat(weightsfile)
	theta = [data['Theta1'],data['Theta2']]
	N = NeuralNetwork(theta=theta)
	print "Test data accurate to %f" % N.get_accuracy(X,y)
	C,G = N.cost_function(X,y,N.roll(theta))
	print "Cost a test Theta %f (should be near .3844)" % C

def training_test(fname):
	data = loadmat(fname)
	X = data['X']
	y = data['y'] - 1
	y = np.array([i[0] for i in y])
	layers = [400,25,10]
	N = NeuralNetwork(layer_sizes=layers)
	N.train(X,y)
	print "Accurate to %f (should be > .93)" % N.get_accuracy(X,y)

"""
trainer interface
"""
def trainer(filename,outfile,layer_sizes):
	data = loadmat(filename)
	X = data['X']
	X = normalize(X)
	layer_sizes.insert(0,X.shape[1])
	y = np.array([i[0] for i in data['y']])
	N = NeuralNetwork(layer_sizes=layer_sizes)
	N.train(X,y)

	thetas = {}
	thetas[LAYERS_KEY] = len(N.theta)
	for i,theta in enumerate(N.theta):
		vname = 'T%d' % i
		thetas[vname] = theta
	savemat(outfile,thetas)
	print "\n\nSaved data to %s." % outfile
	print "Training data self test accurate to %f" % N.get_accuracy(X,y)

"""
predictor interface
"""
def predictor(datafile,thetafile):pass


"""
NeuralNetwork

Keyword Args:

activationFn 	: function eg. sigmoid
activationFnGrad: gradient function eg. sigmoidgradient
layer_sizes	    : nodes in NN in format [NFEATURES, LAYER1...LAYERN, NCLASSES]
theta       	: list of theta matrices eg. theta[0] is theta for input->hidden layer transformation
			  	  used with saved matrices only
"""

class NeuralNetwork:

	def __init__(self,activationFn=sigmoid,activationFnGrad=sigmoidgradient,layer_sizes=None,theta=None):
		self.activationFn = activationFn
		self.activationFnGrad = activationFnGrad
		self.theta = []		
		if theta:
			self.theta = theta
		elif layer_sizes:
			if len(layer_sizes) < 3:
				raise Exception("Need atleast single hidden layer")
			for i in range(0,len(layer_sizes)-1):
				dim = (layer_sizes[i+1],layer_sizes[i]+1)
				self.theta.append(EPSILON * np.random.random_sample(dim))
		else:
			raise Exception("Invalid use must supply either theta or params.")

	"""
	train neural network on provided dataset
	"""
	def train(self,X,y): 
		theta0 = self.roll(self.theta)
		X = normalize(X)
		results = minimizer(lambda x: self.cost_function(X,y,x),theta0,approx_grad = False)
		self.theta = self.unroll(self.theta,results[0])

	"""
	feedforward that returns A & Z as lists
	"""
	def feedforward(self,X,theta):
		nExamples, nFeatures = X.shape

		#weirdness about column vectors
		X = np.concatenate((np.ones((X.shape[0],1)),X),1)
		A = [X[:]]
		Z = [None] #spacer because there is no Z[0]
		
		#compute feed forward propagation
		for W in theta:
			 #add column of ones
			X = X.dot(W.T)
			Z.append(X[:])
			X = self.activationFn(X)
			X = np.concatenate((np.ones((X.shape[0],1)),X),1)
			A.append(X[:])

		A[-1] = A[-1][:,1:] #remove 1's from output layer

		return A,Z

	"""
	return prediction for this set of examples
	"""
	def predict(self,X,theta=None):
		if not theta: theta = self.theta
		A,Z = self.feedforward(X,theta)
		results = np.argmax(A[-1],1)
		return results
	
	"""
	return the accuracy of the current thetas on the data set X
	"""
	def get_accuracy(self,X,y):
		return sum(map(lambda x: 1 if x[0] == x[1] else 0,zip(self.predict(X),y))) / float(len(y))

	"""
	convert vector like [0 1 2] to [[1 0 0][0 1 0][0 0 1]]
	y : original vector
	N : number of classes
	"""
	def bool_matrix_rep(self,y,N):
		I = np.identity(N)
		return np.array([I[i].tolist() for i in y])

	"""
	unroll the provided vector into the format provided by the template
	"""
	def unroll(self,template,vector):
		offset = 0 
		M = template[:]
		for i,W in enumerate(template):
			size = W.shape[0] * W.shape[1]
			partial = vector[offset:offset + size]
			M[i] = np.reshape(partial,W.shape)
			offset += size
		return M

	"""
	roll the provided data (an list of np.arrays) into a vector
	"""
	def roll(self,M):
		return np.concatenate([W.flatten() for W in M])



	"""
	return cost & gradient for this theta
	"""
	def cost_function(self,X,y,flat_theta,lmbd=1.):
		theta = self.unroll(self.theta,flat_theta)
		Y  = self.bool_matrix_rep(y,self.theta[-1].shape[0])
		A,Z = self.feedforward(X,theta)
		H = A[-1]
		m,n = H.shape
		

		#compute cost
		J = (-1 * Y) * np.log(H) - (np.ones(Y.shape) - Y)\
			* np.log(np.ones(H.shape) - H)
		J = 1./m * np.sum(np.sum(J))
		
		#compute regularization term
		R =  lmbd/(2.0 * m) * sum([sum([q**2 for q in np.nditer(layer)]) for layer in theta])
		J += R

		G = self.gradient(A,Z,Y,(m,n),theta)

		if isnan(J):import pdb;pdb.set_trace();
	

		return J,G


	"""	
	calculate gradient using vectorized, back propagation, variable layers
	"""
	def gradient(self,A,Z,Y,shape,theta):
		m,n = shape
		D = [None]  * len(theta)
		D[-1] = (A[-1] - Y)# for the output layer
		for i in range(len(theta)-2,-1,-1):
			D[i] = D[i+1].dot(theta[i+1])
			D[i] = D[i][:,1:]
			D[i] = D[i] * self.activationFnGrad(Z[i+1])
			
		G = [d.T.dot(A[i])/m for i,d in enumerate(D)]

		return self.roll(G)



