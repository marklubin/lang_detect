"""
Defined Constants etc.
"""

from os import path,environ

LANGS = ["English","Russian","French"]

EPSILON = .12

"""
various data files, directories
"""

TRAINING_TEST_FILE = path.abspath(path.join(".","testing","ex3data1.mat"))
WEIGHTS_TEST_FILE = path.abspath(path.join(".","testing","ex3weights.mat"))
CLASSIFIERS_DIR = path.abspath(path.join(".","classifiers"))
AUDIO_DIR = path.abspath(path.join(".","audio"))
FEATURES_DIR = path.abspath(path.join(".","features"))

"""
SMILExtract configuration
"""
CONF_BASE = environ['SMILE_CONF']
CONF_RECORD_FILE = path.join("demo","audiorecorder.conf")
CONF_FILE = "emo_IS09.conf"

"""
mat file keys
"""
LAYERS_KEY = 'NLAYERS'
MEAN_KEY = 'MEAN'
STD_KEY = 'STD'

"""
dummy files
"""
FEATURES_FILE = "FEATS"
WAVE_FILE = "out.wav"