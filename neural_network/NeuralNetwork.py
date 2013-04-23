"""
	NeuralNetwork.py
	Single Hidden Layer(for now) MLP Neural Network
	Mark Lubin
"""

"""
TODO:
Handle unroll/rolling of thetas
implement gradient
"""
ITERS = 20
import numpy as np
from scipy.io import loadmat
from scipy.optimize import fmin_tnc as minimizer

class NeuralNetwork:
	"""
	activationFn: function eg. g = sigmoid(x)
	description : what is this NN? eg. English V. Chinese Classifier
	nUnits      : size of hidden layer
	nClasses    : number of output classes
	theta       : list of theta matrices eg. theta[0] is theta for input->hidden layer transformation
	"""

	def __init__(self,activationFn,activationFnGrad,nUnits,nClasses,inputSize,nFeatures,description,theta=None):
		self.activationFn = activationFn
		self.activationFnGrad = activationFnGrad
		self.description = description
		#self.nLayers = nLayers TODO variable number of hidden layers
		self.nUnits = nUnits
		self.nClasses = nClasses
		self.nFeatures = nFeatures
		if theta:
			self.theta = theta
		else:
			self.theta = [.12 * np.random.rand(nUnits,nFeatures+1), .12 * np.random.rand(nClasses,nUnits+1)]

	"""
	train neural network on provided dataset
	"""
	def train(self,X,y):
		#import pdb;pdb.set_trace();
		theta0 = self.roll(self.theta)
		results = minimizer(lambda x: self.cost_function(X,y,x),theta0,lambda x: self.gradient(X,y,x))
		self.theta = self.unroll(self.theta,results[0])
		

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
	unroll the provided vector into the format provided by the template
	"""
	@staticmethod
	def unroll(template,vector):
		#unroll theta 
		offset = 0 
		M = template[:]
		for i,W in enumerate(template):
			size = W.shape[0] * W.shape[1]
			partial = vector[offset:offset + size]
			M[i] = np.reshape(partial,W.shape)
			offset += size
		return M

	"""
	roll the provided data (an array of npdarrays) in a vector
	"""
	@staticmethod
	def roll(M):
		return np.concatenate([W.flatten() for W in M])



	"""
	return gradient for this matrix 
	"""
	def cost_function(self,X,y,flat_theta,lmbd=1.):
		#import pdb;pdb.set_trace();
		theta = self.unroll(self.theta,flat_theta)
		Y  = self.matrix_rep(y,self.nClasses)
		H = self.feedforward(X,theta)
		m,n = H.shape
		

		#compute cost
		J = (-1 * Y) * np.log(H) - (np.ones(Y.shape) - Y)\
			* np.log(np.ones(H.shape) - H)
		J = 1./m * np.sum(np.sum(J))
		
		#compute regularization term
		R =  lmbd/(2.0 * m) * sum([sum([q**2 for q in np.nditer(layer)]) for layer in theta])

		print "Cost: %f" % (J+R)
		return J + R


	def gradient(self,X,y,flat_theta):
		m,n = X.shape
		theta = self.unroll(self.theta,flat_theta)
		Y = self.matrix_rep(y,self.nClasses)

		delta1 = np.zeros(theta[0].shape)
		delta2 = np.zeros(theta[1].shape)

		for i,row in enumerate(X):
			a1 = np.concatenate(([1],row))
			z2 = a1.dot(theta[0].T)
			a2 = np.concatenate(([1],self.activationFn(z2)))
			z3 = a2.dot(theta[1].T)
			a3 = self.activationFn(z3)

			yi = Y[i]

			d3 = a3 - yi

			d2 = d3.dot(theta[1]) * np.concatenate(([1],self.activationFnGrad(z2)))
			d2 = d2[1:]
			d3 = np.reshape(d3,(d3.shape[0],1))
			a2 = np.reshape(a2,(1,a2.shape[0]))
			
			d2 = np.reshape(d2,(d2.shape[0],1))
			a1 = np.reshape(a1,(1,a1.shape[0]))
			

			delta2 += d3.dot(a2)
			delta1 += d2.dot(a1)

		theta1grad = delta1/m
		theta2grad = delta2/m

		return self.roll([theta1grad,theta2grad])

def sigmoid(X):
	return 1.0 / (1.0 + np.exp(-1. * X))

def sigmoidgradient(z):
	s = sigmoid(z)
	return s * (np.ones(z.shape) - s)

def main():
	data = loadmat('ex3data1.mat')
	X = data['X']

	y = data['y'] - 1
	y = np.array([i[0] for i in y])
	data = loadmat('ex3weights.mat')
	theta = [data['Theta1'],data['Theta2']]
	N = NeuralNetwork(sigmoid,sigmoidgradient,25,10,X.shape[0],X.shape[1],"ex3test",theta)
	print "Test data accurate to %f" % N.getAccuracy(X,y)
	print "Cost a test Theta %f" % N.cost_function(X,y,N.roll(theta))
	print N.gradient(X,y,N.roll(theta))

def prop_test():
	data = loadmat('ex3data1.mat')
	X = data['X']
	y = data['y'] - 1
	y = np.array([i[0] for i in y])
	N = NeuralNetwork(sigmoid,sigmoidgradient,25,10,X.shape[0],X.shape[1],"ex3test")
	N.train(X,y)
	print "Test data accurate to %f" % N.getAccuracy(X,y)

if __name__ == "__main__": prop_test()


