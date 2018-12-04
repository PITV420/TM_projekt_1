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
        if not cfg['covariance_type'] == 'diag' and \
                not cfg['covariance_type'] == 'full' and \
                not cfg['covariance_type'] == 'tied' and \
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


def save(obj, name):
    file = open(name, 'wb')
    pickle.dump(obj, file)

for i in range(1, 24):

    config = loadConfig('config/gmm_config/gmm_' + str(i) + '.cfg')
    for k in range(1, 24):
        parametrized_data = loadData('files/test_mfcc_mod/parametrized_' + str(k) + '.p')
        data_ = eachDigitGMM(parametrized_data, config)
        save(data_, 'files/gmm_models/digits_gmm_'+str(i)+'_'+str(k)+'.p')
