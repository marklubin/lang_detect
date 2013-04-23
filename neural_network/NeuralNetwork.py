"""
	NeuralNetwork.py
	Single Hidden Layer(for now) MLP Neural Network
	Mark Lubin
"""
EPSILON = .12
import numpy as np
from scipy.io import loadmat,savemat
from scipy.optimize import fmin_tnc as minimizer

class NeuralNetwork:
	"""
	activationFn: function eg. g = sigmoid(x)
	description : what is this NN? eg. English V. Chinese Classifier
	params		: list of matrix sizes for the thetas
	theta       : list of theta matrices eg. theta[0] is theta for input->hidden layer transformation
				  used with saved matrices only
	"""

	def __init__(self,activationFn,activationFnGrad,params,description,theta=None):
		self.activationFn = activationFn
		self.activationFnGrad = activationFnGrad
		self.description = description
		self.theta = []		
		if theta:
			self.theta = theta
		else:
			for dim in params:
				self.theta.append(EPSILON * np.random.random_sample(dim))
			#import pdb;pdb.set_trace();
			#self.theta = [.12 * np.random.rand(nUnits,nFeatures+1), .12 * np.random.rand(nClasses,nUnits+1)]

	"""
	train neural network on provided dataset
	"""
	def train(self,X,y):
		#import pdb;pdb.set_trace();
		theta0 = self.roll(self.theta)
		results = minimizer(lambda x: self.cost_function(X,y,x),theta0,approx_grad = False)
		#,lambda x: self.gradient(X,y,x))
		self.theta = self.unroll(self.theta,results[0])

	"""
	feedforward that returns A & Z as lists
	"""
	def feedforward(self,X,theta):
		nExamples, nFeatures = X.shape
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
		return np.argmax(A[-1],1)
	
	"""
	return the accuracy of the current thetas on the data set X
	"""
	def getAccuracy(self,X,y):
		return sum(map(lambda x: 1 if x[0] == x[1] else 0,zip(self.predict(X),y))) / float(len(y))

	"""
	convert vector like [0 1 2] to [[1 0 0][0 1 0][0 0 1]]
	y : original vector
	N : number of classes
	"""
	def matrix_rep(self,y,N):
		I = np.identity(N)
		return np.array([I[i].tolist() for i in y])

	"""
	unroll the provided vector into the format provided by the template
	"""
	def unroll(self,template,vector):
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
	def roll(self,M):
		return np.concatenate([W.flatten() for W in M])



	"""
	return cost & gradient for this theta
	"""
	def cost_function(self,X,y,flat_theta,lmbd=1.):
		#import pdb;pdb.set_trace();
		theta = self.unroll(self.theta,flat_theta)
		Y  = self.matrix_rep(y,self.theta[-1].shape[0])
		A,Z = self.feedforward(X,theta)
		H = A[-1]
		m,n = H.shape
		

		#compute cost
		J = (-1 * Y) * np.log(H) - (np.ones(Y.shape) - Y)\
			* np.log(np.ones(H.shape) - H)
		J = 1./m * np.sum(np.sum(J))
		
		#compute regularization term
		R =  lmbd/(2.0 * m) * sum([sum([q**2 for q in np.nditer(layer)]) for layer in theta])

		print "Cost: %f" % (J+R)

		G = self.gradient(A,Z,Y,(m,n),theta)
		#G1 = self.gradient(X,y,flat_theta)
		#import pdb;pdb.set_trace();
		return (J + R),G


	"""
	calculate gradient using vectored, back propagation, variable layers
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
	params = [(25,401),(25,26),(10,26)]
	N = NeuralNetwork(sigmoid,sigmoidgradient,params,"ex3test",theta)
	print "Test data accurate to %f" % N.getAccuracy(X,y)
	C,G = N.cost_function(X,y,N.roll(theta))
	print "Cost a test Theta %f" % C
	print "Gradient: " + str(G)

def prop_test():
	data = loadmat('ex3data1.mat')
	X = data['X']
	y = data['y'] - 1
	y = np.array([i[0] for i in y])
	params = [(25,401),(10,26),(10,11),(10,11)]
	N = NeuralNetwork(sigmoid,sigmoidgradient,params,"ex3test")
	N.train(X,y)
	print "Test data accurate to %f" % N.getAccuracy(X,y)

if __name__ == "__main__": prop_test()


