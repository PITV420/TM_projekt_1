from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import numpy as np
import pickle
import collections


def loadData(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


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
        cfg['tolerance'] = float(cfg['tolerance'])
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
            'tolerance': 0.001,
            'covariance_type': 'diag',
        }

    return cfg


def eachDigitGMM(data, cfg):
    """ Compute GMM models for each digit """

    """
    takes:
        data - matrix of MFCC parametrized vocal samples.
            rows - following spoken numbers
            columns - following speakers
        cfg - config file for GMM estimator
        
    returns:
        Dictionary of GMM models, where key is the label of a digit and value is a GMM model
    """

    models = {}
    for j in range(len(data)):
        train_set = data[j][0]
        for i in range(1, len(data[j])):
            train_set = np.concatenate((train_set, data[j][i]), axis=0)

        estimator = GaussianMixture(n_components=cfg['components'], max_iter=cfg['max_iterations'],
                                    tol=cfg['tolerance'], covariance_type=cfg['covariance_type'])
        models[j] = estimator.fit(train_set)

    return models


def save(obj):
    file = open('files/digits_gmm.p', 'wb')
    pickle.dump(obj, file)


parametrized_data = loadData('files/parametrized_delta_delta.p')
config = loadConfig('config/gmm.cfg')

data_ = eachDigitGMM(parametrized_data, config)

save(data_)


