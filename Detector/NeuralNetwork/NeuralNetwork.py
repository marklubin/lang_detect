"""
NeuralNetwork.py
MLP Neural Network with support for any number of hidden layers

NeuralNetwork Keyword Args:

activationFn 	: function eg. sigmoid
activationFnGrad: gradient function eg. sigmoidgradient
layer_sizes	    : nodes in NN in format [NFEATURES, LAYER1...LAYERN, NCLASSES]
theta       	: list of theta matrices eg. theta[0] is theta for input->hidden layer transformation
mean 			: mean for normalization transformation
std             : std for normalization  transformation
lmbda           : regularization parameter

Mark Lubin
"""
from ..configuration import *
from math import isnan
from normalizer import normalize,normalize2,normalize3
from scipy.optimize import fmin_tnc as minimizer
from .utils import *

class NeuralNetwork:

	def __init__(self,activationFn=sigmoid,activationFnGrad=sigmoidgradient,layer_sizes=None,theta=None,\
					mean=None,std=None,lmbda=1):
		self.activationFn = activationFn
		self.activationFnGrad = activationFnGrad
		self.theta = []	
		self.lmbda = lmbda
		self.mean = mean
		self.std = std	
		self.nTrainingExamples = 0
		if theta:
			self.theta = theta
		elif layer_sizes:
			if len(layer_sizes) < 3:
				raise Exception("Need atleast single hidden layer")
			for i in range(0,len(layer_sizes)-1):
				dim = (layer_sizes[i+1],layer_sizes[i]+1)
				self.theta.append(EPSILON * np.random.random_sample(dim))

	"""
	train neural network on provided dataset
	"""
	def train(self,X,y): 
		theta0 = self.roll(self.theta)
		X,self.mean,self.std = normalize(X)
		self.nTrainingExamples = X.shape[0]
		results = minimizer(lambda x: self.cost_function(X,y,x),theta0,approx_grad = False)
		self.theta = self.unroll(self.theta,results[0])
		return results

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
		if self.mean is not None and self.std is not None:
			#handle python row/col vector weirdness
			if X.shape[0] == 1:
				normalize2(X,self.mean,self.std)
			else:
				X = normalize3(X,self.mean,self.std)

		A,Z = self.feedforward(X,theta)
		results = np.argmax(A[-1],1)
		return results
	
	"""
	return the accuracy of the current thetas on the data set X
	"""
	def get_accuracy(self,X,y):
		predictions = self.predict(X)
		return sum(map(lambda x: 1 if x[0] == x[1] else 0,zip(predictions,y))) / float(len(y)),predictions

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
	def cost_function(self,X,y,flat_theta):
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
		R =  self.lmbda/(2.0 * m) * sum([sum([q**2 for q in np.nditer(layer)]) for layer in theta])
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

	"""
	save NeuralNetwork params to file
	"""
	def save(self,filename):
		data_dict = {}
		data_dict[MEAN_KEY] = self.mean
		data_dict[STD_KEY] = self.std
		data_dict[LAYERS_KEY] = len(self.theta)
		data_dict[LMBDA_KEY] = self.lmbda
		data_dict[NTRAININGEXAMPLES_KEY] = self.nTrainingExamples
		for i,theta in enumerate(self.theta):
			vname = THETA_KEY_FORMAT_STR % i
			data_dict[vname] = theta
		savemat(filename,data_dict)

	"""
	load NeuralNetwork params from filename
	"""
	def load(self,filename):
		theta = []
		data = loadmat(filename)
		self.mean = data[MEAN_KEY]
		self.std = data[STD_KEY]
		self.lmbda = data[LMBDA_KEY][0][0]
		self.nTrainingExamples = data[NTRAININGEXAMPLES_KEY] 
		nThetas = data[LAYERS_KEY]
		for i in range(0,nThetas):
			theta_name = THETA_KEY_FORMAT_STR % i
			self.theta.append(data[theta_name])



