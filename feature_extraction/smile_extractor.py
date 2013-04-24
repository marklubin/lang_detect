"""
smile_extractor.py
-append the feature matrices with result of running extactor on directory
-call as 
	python smile_extractor.py <DIRECTORY> <LABEL> <MAT_FILE>

Mark Lubin

"""
from scipy.io import loadmat,savemat
import numpy as np
from sys import argv
from os import system,listdir,unlink,environ

CONF_BASE = environ['SMILE_CONF'] + "\\"
CONF_FILE = "emo_IS09.conf"
FEATURES_FILE = "FEATS"

def main():
	directory = argv[1]
	label = int(argv[2])
	outfilename = argv[3]
	
	try:
		 data = loadmat(outfilename)
		 X = data['X'].tolist()
		 y = [x[0] for x in data['y'].tolist()]
	except IOError:
		 X = []
		 y = []

	for fname in listdir(directory):
		features = parse_features(directory + "\\" + fname)
		X.append(features)
		y.append(label)

	X = normalized(X)
	results = {'X' : np.array(X),'y' :y}

	savemat(outfilename,results)

def parse_features(fname):
	system("SMILExtract -C %s -I %s -O %s -noconsoleoutput -l 0" \
		%(CONF_BASE + CONF_FILE, fname,FEATURES_FILE))
	f = open(FEATURES_FILE,"r")
	while f.readline() != "@data\n":pass
	f.readline()
	features = [float(x) for x in f.readline().split(',')[1:-1]]
	f.close()
	unlink(FEATURES_FILE)

	return features

def normalized(X):
	return X
	


if __name__ == '__main__':main()