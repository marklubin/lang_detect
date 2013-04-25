"""
normalizer.py
normalize feature set

Peter Potash
"""
import numpy as np

def normalize(X):
	meanVector = np.mean(X, axis=0)
	stdVector = np.std(X, axis=0)
	for comp in stdVector:
		if comp == 0:
			comp = 1
	meanMatrix = np.kron(np.ones((X.shape[0], 1)), meanVector)
	stdMatrix = np.kron(np.ones((X.shape[0], 1)), stdVector)
	return (X-meanMatrix)/stdMatrix