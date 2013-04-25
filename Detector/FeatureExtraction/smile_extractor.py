from scipy.io import loadmat,savemat
from os import system,unlink,environ,path
from ..configuration import *

def parse_features(fname):
	system("SMILExtract -C %s -I %s -O %s -noconsoleoutput" \
		%(path.join(CONF_BASE,CONF_FILE), fname,FEATURES_FILE))
	f = open(FEATURES_FILE,"r")
	while f.readline() != "@data\n":pass
	f.readline()
	features = [float(x) for x in f.readline().split(',')[1:-1]]
	f.close()
	unlink(FEATURES_FILE)

	return features
	


if __name__ == '__main__':main()