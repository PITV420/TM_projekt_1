from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
import pickle


def loadConfig(path):
    try:
        file = open(path, 'r')
        lines = file.readlines()
        file.close()

        cfg = {}
        for line in lines:
            key, value = line.replace('\n', '').split('=')
            cfg[key] = value

        cfg['components'] = int(cfg['components'])
        cfg['max_iterations'] = int(cfg['max_iterations'])
        cfg['toleration'] = float(cfg['toleration'])
        if not cfg['covariance_type'] == 'diag' and\
           not cfg['covariance_type'] == 'full' and\
           not cfg['covariance_type'] == 'tied' and\
           not cfg['covariance_type'] == 'spherical':
            cfg['covariance_type'] = 'diag'

    except Exception as e:
        print('Error:', e, '// using default config')
        cfg = {
            'components': 8,
            'max_iterations': 30,
            'toleration': 0.001,
            'covariance_type': 'diag',
        }

    return cfg


def loadData(pathMFCC, pathMFCC_labels):
    fileMFCC = open(pathMFCC, 'rb')
    fileMFCC_labels = open(pathMFCC_labels, 'rb')
    return pickle.load(fileMFCC), pickle.load(fileMFCC_labels)


def trainDigit(data, cfg):
    """ Input data must be just one digit! """

    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(data):
        print(train_index, test_index)


config = loadConfig('config/gmm.cfg')
MFCC, MFCC_labels = loadData('files/parametrized.p', 'files/mfcc_matrix_scheme.p')
trainDigit(MFCC_labels[0], config)
