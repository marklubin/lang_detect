"""
Defined Constants etc.
"""

from os import path,environ
import numpy as np
from scipy.io import loadmat,savemat

LANGS = ["English","Russian","French","Mandarin","Japanese"]

NEURAL_MODES = ['train','predict']
EPSILON = .12

TRAIN = 'train'
PREDICT = 'predict'

"""
various data files, directories
"""

CLASSIFIER_LOG_FILE = path.abspath(path.join(".","testing","classifier.csv"))
CLASSIFIER_LOG_FILE_HEADER = "testdatafile,classifierfile,languages,featureset,thetashapes,nTrainingExamples,nTestExamples,nFeatures,nLabels,cost,accuracy,lmbda\n"
TRAINING_TEST_FILE = path.abspath(path.join(".","testing","ex3data1.mat"))
WEIGHTS_TEST_FILE = path.abspath(path.join(".","testing","ex3weights.mat"))
CLASSIFIERS_DIR = path.abspath(path.join(".","classifiers"))
AUDIO_DIR = path.abspath(path.join(".","audio"))
FEATURES_DIR = path.abspath(path.join(".","features"))
TESTING_DIR = path.abspath(path.join(".","testing"))

"""
SMILExtract configuration
"""
CONF_BASE = environ['SMILE_CONF']
CONF_RECORD_FILE = path.join("demo","audiorecorder.conf")
CONF_FILE = "emo_IS09.conf"
SMILE_CALL = "SMILExtract -C %s -I \"%s\" -O \"%s\" -noconsoleoutput"

"""
mat file keys
"""
LAYERS_KEY = 'NLAYERS'
MEAN_KEY = 'MEAN'
STD_KEY = 'STD'
LMBDA_KEY = 'LMBDA'
X_KEY = 'X'
Y_KEY = 'y'
NLABELS_KEY = 'nLabels'
FEATURE_SET_KEY = 'features'
LANGS_KEY = 'languages'
THETA_KEY_FORMAT_STR = 'T%d'
Y_ACTUAL_KEY = 'yActual'
Y_PREDICTED_KEY = 'yPredicted'
NTRAININGEXAMPLES_KEY = 'nTrainingExamples'

"""
dummy files
"""
FEATURES_FILE = "features.arff"
WAVE_FILE = "out.wav"