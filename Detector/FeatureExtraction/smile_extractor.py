"""
smile_extractor.py
extract features from SMILExtractor using ARFF output format

Mark Lubin
"""

from os import system,unlink,environ,path
from ..configuration import *

def parse_features(fname,smile_call,configfile):
	system(smile_call %(path.join(CONF_BASE,configfile), fname,FEATURES_FILE))
	f = open(FEATURES_FILE,"r")
	while f.readline() != "@data\n":pass
	f.readline()
	features = [float(x) for x in f.readline().split(',')[1:-1]]
	f.close()
	unlink(FEATURES_FILE)

	return features
	


if __name__ == '__main__':main()