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


def validateDigit(data, cfg):
    kf = KFold(n_splits=5, shuffle=False, random_state=None)
    """ Split data into separate digits """
    for digit in data:
        """ Get indexes for train-test sets """
        for train_index, test_index in kf.split(digit):
            """ Concatenate train data into one matrix """
            train_set = digit[train_index[0]]
            for index in train_index:
                if not index == train_index[0]:
                    train_set = np.concatenate((train_set, digit[index]), axis=0)
            """ For each number generate unique estimator! """
            print(train_set.shape)


config = loadConfig('config/gmm.cfg')
MFCC, MFCC_labels = loadData('files/parametrized.p', 'files/mfcc_matrix_scheme.p')
MFCC, MFCC_labels = np.asarray(MFCC), np.asarray(MFCC_labels)
validateDigit(MFCC, config)
