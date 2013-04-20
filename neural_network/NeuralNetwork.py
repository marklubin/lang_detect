"""
	NeuralNetwork.py
	Single Hidden Layer(for now) MLP Neural Network
	Mark Lubin
"""

class NeuralNetwork:
	"""
	activationFn: function eg. g = lambda x :sigmoid(x)
	description : what is this NN? eg. English V. Chinese Classifier
	nUnits      : size of hidden layer
	nClasses    : number of output classes
	theta       : list of theta vectors eg. theta[0] is theta for input->hidden layer transformation
	"""

	def __init__(self,activationFn,description,nUnits,nClasses):
		self.activationFn = activationFn
		self.description = description
		self.nUnits = nUnits
		self.nClasses = nClasses
		#self.theta = [rand_weight_init(THETA1SIZE), rand_weight_init(THETA2SIZE)]

	"""
	train neural network on provided dataset
	"""
	def train(self,trainingdata):pass

	"""
	predict class for example
	"""
	def predict(self,features):pass

	"""
	return a random matrix of requested size
	"""
	def rand_weight_init(size):pass


	"""
	return gradient for this training example
	"""
	def cost_function(X,y):pass

	"""
	string representation of NeuralNetwork
	"""
	def __str__(self):pass

	"""
	serialized version of NeuralNetwork
	"""
	def __repr__(self):pass