"""
Neural Network Utility Functions

Mark Lubin
"""
import NeuralNetwork as NN
from ..configuration import *
from random import sample

"""
default activation function and gradient
"""
def sigmoid(X):
	return 1.0 / (1.0 + np.exp(-1. * X))

def sigmoidgradient(z):
	s = sigmoid(z)
	return s * (np.ones(z.shape) - s)

"""
trainer interface
"""
def trainer(filename,outfile,layer_sizes,lmbda):
	data = loadmat(filename)
	X = data[X_KEY]
	nLabels = data[NLABELS_KEY]
	layer_sizes.insert(0,X.shape[1])
	layer_sizes.append(nLabels)
	y = np.array([i[0] for i in data[Y_KEY]])
	N = NN.NeuralNetwork(layer_sizes=layer_sizes,lmbda=lmbda)
	N.train(X,y)
	N.save(outfile)
	print "\n\nSaved data to %s." % outfile
	print "Training data self test accurate to %f" % N.get_accuracy(X,y)[0]


"""
subset_trainer: train on a subset of the training data
"""
def subset_trainer(filename,outfile,layer_sizes,lmbda,sample_size):
	data = loadmat(filename)
	X = data[X_KEY]
	Y = data[Y_KEY]
	nLabels = data[NLABELS_KEY]
	layer_sizes.insert(0,X.shape[1])
	layer_sizes.append(nLabels)

	if sample_size >= X.shape[0]:
		errormsg = """Sample size bigger than data available. Requested %d, Found %d.""" \
					%(sample_size,X.shape[0])
		raise Exception(errormsg)

	indicies = sample(range(0,X.shape[0]),sample_size)
	xSample = []
	ySample = []

	for i in indicies:
		xSample.append(X[i])
		ySample.append(Y[i][0])

	xSample = np.array(xSample)
	N = NN.NeuralNetwork(layer_sizes=layer_sizes,lmbda=lmbda)
	N.train(xSample,ySample)
	N.save(outfile)
	print "\n\nSaved data to %s." % outfile
	print "Training data self test accurate to %f" % N.get_accuracy(xSample,ySample)[0]




"""
predictor interface
"""
def predictor(datafile,thetafile,logfile,resultsfile):

	#get all the data
	data = loadmat(datafile)
	X = data[X_KEY]
	y = np.array([i[0] for i in data[Y_KEY]])
	nLabels = data[NLABELS_KEY]
	languages = data[LANGS_KEY][0]
	feature_set = data[FEATURE_SET_KEY][0]
	N = NN.NeuralNetwork()
	N.load(thetafile)
	nTrainingExamples = N.nTrainingExamples
	cost,grad = N.cost_function(X,y,N.roll(N.theta))
	accuracy,predictions = N.get_accuracy(X,y)
	lmbda = N.lmbda
	thetashapes = str([THETA.shape for THETA in N.theta])
	loglinedata = (datafile,thetafile,languages,feature_set,thetashapes,nTrainingExamples,\
		X.shape[0],X.shape[1],nLabels,cost,accuracy,lmbda)
	results = {Y_ACTUAL_KEY : y, Y_PREDICTED_KEY : predictions}
	savemat(resultsfile,results)
	logger(logfile,loglinedata)

"""
log prediction results
"""
def logger(logfile,loglinedata):
	try:
		f = open(logfile,'r')
		f.close()
	except IOError:
		f = open(logfile,'w')
		f.write(CLASSIFIER_LOG_FILE_HEADER)
		f.close()
	f = open(logfile,'a')
	logline = "%s,%s,%s,%s,%s,%d,%d,%d,%d,%f,%f,%f\n" % loglinedata
	f.write(logline)	
	print "\nWrote prediction results to %s, accuracy %.4f" % (logfile,loglinedata[-2])
	f.close()
