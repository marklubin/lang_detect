"""
	NeuralNetwork.py
	Single Hidden Layer(for now) MLP Neural Network
	Mark Lubin
"""
import numpy as np
from scipy.io import loadmat
from scipy.optimize import fmin_cg

class NeuralNetwork:
	"""
	activationFn: function eg. g = sigmoid(x)
	description : what is this NN? eg. English V. Chinese Classifier
	nUnits      : size of hidden layer
	nClasses    : number of output classes
	theta       : list of theta matrices eg. theta[0] is theta for input->hidden layer transformation
	"""

	def __init__(self,activationFn,nUnits,nClasses,inputSize,nFeatures,description,theta=None):
		self.activationFn = activationFn
		self.description = description
		#self.nLayers = nLayers TODO variable number of hidden layers
		self.nUnits = nUnits
		self.nClasses = nClasses
		self.nFeatures = nFeatures
		if theta:
			self.theta = theta
		else:
			self.theta = [np.random.rand(nUnits,nFeatures+1), np.random.rand(nClasses,nUnits+1)]

	"""
	train neural network on provided dataset
	"""
	def train(self,X,y):pass

	"""
	feedforward propagation algorithm
	"""
	def feedforward(self,X,theta):
		nExamples, nFeatures = X.shape
		if(nFeatures != self.nFeatures):
			raise Exception("Invalid number of features got %d expected %d." % (nFeatures, self.nFeatures) )
		
		#compute feed forward propagation
		for W in theta:
			#import pdb;pdb.set_trace();
			X = np.concatenate((np.ones((X.shape[0],1)),X),1) #add column of ones
			X = self.activationFn(X.dot(W.T))

		return X

	"""
	return prediction for this set of examples
	"""
	def predict(self,X,theta=None):
		if not theta: theta = self.theta
		return np.argmax(self.feedforward(X,theta),1)
	
	"""
	return the accuracy of the current thetas on the data set X
	"""
	def getAccuracy(self,X,y):
		return sum(map(lambda x: 1 if x[0] == x[1] else 0,zip(self.predict(X),y))) / float(len(y))

	"""
	convert vector like [0 1 2] to [[1 0 0]
									[0 1 0]
									[0 0 1]]
	y : original vector
	N : number of classes
	"""
	def matrix_rep(self,y,N):
		I = np.identity(N)
		return np.array([I[i].tolist() for i in y])


	"""
	return gradient for this matrix 
	"""
	def cost_function(self,X,y,theta,lmbd=1.):
		Y  = self.matrix_rep(y,self.nClasses)
		H = self.feedforward(X,theta)
		m,n = H.shape

		#compute cost
		J = (-1 * Y) * np.log(H) - (np.ones(Y.shape) - Y)\
			* np.log(np.ones(H.shape) - H)
		J = 1./m * np.sum(np.sum(J))
		
		#compute regularization term
		R =  lmbd/(2.0 * m) * sum([sum([q**2 for q in np.nditer(layer)]) for layer in theta])

		return J + R

	"""
	compute gradient at this theta
	"""
	def gradient(self,X,y,theta):pass



	"""
	string representation of NeuralNetwork
	"""
	def __str__(self):pass

	"""
	serialized version of NeuralNetwork, probably just need to store thetas
	"""
	def __repr__(self):pass

def sigmoid(X): 
	return 1.0 / (1.0 + np.exp(-1. * X))

def main():
	data = loadmat('ex3data1.mat')
	X = data['X']

	y = data['y'] - 1
	y = np.array([i[0] for i in y])
	data = loadmat('ex3weights.mat')
	theta = [data['Theta1'],data['Theta2']]
	N = NeuralNetwork(sigmoid,25,10,X.shape[0],X.shape[1],"ex3test",theta)
	print "Test data accurate to %f" % N.getAccuracy(X,y)
	print "Cost a test Theta %f" % N.cost_function(X,y,theta)

if __name__ == "__main__": main()