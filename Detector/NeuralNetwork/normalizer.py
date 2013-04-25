"""
normalizer.py
normalize feature set

Peter Potash
"""
import numpy as np



def normalize(X):
	meanVector = np.mean(X, axis=0)
	stdVector = np.std(X, axis=0)
	for i,comp in enumerate(stdVector):
		if comp == 0:
			stdVector[i] = 1.
	meanMatrix = np.kron(np.ones((X.shape[0], 1)), meanVector)
	stdMatrix = np.kron(np.ones((X.shape[0], 1)), stdVector)
	return (X-meanMatrix)/stdMatrix,meanVector,stdVector

def normalize2(X,meanVector,stdVector):
	meanVector = np.reshape(meanVector,(1,X.shape[1]))
	stdVector = np.reshape(stdVector,(1,X.shape[1]))
	return np.divide((X-meanVector),stdVector)

def normalize3(X,meanVector,stdVector):
	meanMatrix = np.kron(np.ones((X.shape[0], 1)), meanVector)
	stdMatrix = np.kron(np.ones((X.shape[0], 1)), stdVector)
	return (X-meanMatrix)/stdMatrix