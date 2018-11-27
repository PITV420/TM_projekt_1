from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import numpy as np
import pickle
import collections


def loadData(path):
    file = open(path, 'rb')
    return pickle.load(file)


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
        if not cfg['covariance_type'] == 'diag' or\
           not cfg['covariance_type'] == 'full' or\
           not cfg['covariance_type'] == 'tied' or\
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


def compute_gmm(data, cfg):
    return GaussianMixture(n_components=cfg['components'], covariance_type=cfg['covariance_type'],
                               max_iter=cfg['max_iterations'], tol=cfg['toleration']).fit(data)


parametrized_data = loadData('files/parametrized.p')
config = loadConfig('config/gmm.cfg')


def eachDigitGMM(data, cfg):
    data_mfcc = {}
    aux_mfcc = {}
    for key1 in data:
        for key2 in data[key1]:
            if key2 == list(data[key1].keys())[0]:
                aux_mfcc = data[key1][key2]
            elif key2 > list(data[key1].keys())[0]:
                aux_mfcc = np.concatenate((aux_mfcc, data[key1][key2]), axis=0)
        data_mfcc[key1] = compute_gmm(aux_mfcc, cfg)

    return data_mfcc

def save(obj):
    file = open('files/digits_gmm.p', 'wb')
    pickle.dump(obj, file)


data = eachDigitGMM(parametrized_data, config)

save(data)


