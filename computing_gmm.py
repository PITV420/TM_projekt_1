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


def compute_gmm(data_mfcc, cfg):
    return GaussianMixture(n_components=cfg['components'], covariance_type=cfg['covariance_type'],
                               max_iter=cfg['max_iterations'], tol=cfg['toleration']).fit(data_mfcc)

"""
    gauss_data = 0
    
    means_data = gmm_data.means_[:, feature]
    weights_data = gmm_data.weights_
    covs_data = gmm_data.covariances[:, feature]

    left = min(data_mfcc[:, feature])
    right = max(data_mfcc[:, feature])
    x = np.arange(left, right, 0.001)

    for i in range(len(means_data)):
        gauss_data = gauss_data + norm.pdf(x, means_data[i], covs_data[i]) * weights_data[i]

    return gauss_data
"""


data = loadData('files/parametrized.p')
config = loadConfig('config/gmm.cfg')

gmm_models = {}
for key in data:
    gmm_models[key] = compute_gmm(data[key][0], config)
gmm_models = collections.OrderedDict(sorted(gmm_models.items()))
