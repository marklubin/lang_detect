"""
Neural Network Utility Functions

Mark Lubin
"""
import NeuralNetwork as NN
from ..configuration import *

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
	print "Training data self test accurate to %f" % N.get_accuracy(X,y)

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
	cost,grad = N.cost_function(X,y,N.roll(N.theta))
	accuracy,predictions = N.get_accuracy(X,y)
	lmbda = N.lmbda
	loglinedata = (datafile,thetafile,languages,feature_set,\
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
	logline = "%s,%s,%s,%s,%d,%d,%d,%f,%f,%f\n" % loglinedata
	f.write(logline)	
	print "\nWrote prediction results to %s, accuracy %.4f" % (logfile,loglinedata[-2])
	f.close()
