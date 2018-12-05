"""
Train GMM models for each digit based on training data
"""


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
    """
    :param path: path to config file
    :return: config dictionary
    """

    try:
        with open(path, 'r') as file:
            lines = file.readlines()
            file.close()

            cfg = {}
            for line in lines:
                key, value = line.replace('\n', '').split('=')
                cfg[key] = value

            cfg['window_length'] = float(cfg['window_length'])
            cfg['window_step'] = float(cfg['window_step'])
            cfg['cepstrum_number'] = int(cfg['cepstrum_number'])
            cfg['filter_number'] = int(cfg['filter_number'])
            cfg['preemphasis_filter'] = float(cfg['preemphasis_filter'])
            cfg['use_delta'] = bool(cfg['use_delta'])
            cfg['delta_sample'] = int(cfg['delta_sample'])
            cfg['use_delta_delta'] = bool(cfg['use_delta_delta'])
            cfg['delta_delta_sample'] = int(cfg['delta_delta_sample'])
            if cfg['window_function'] == 'bartlett':
                cfg['window_function'] = np.bartlett
            elif cfg['window_function'] == 'blackman':
                cfg['window_function'] = np.blackman
            elif cfg['window_function'] == 'hanning':
                cfg['window_function'] = np.hanning
            elif cfg['window_function'] == 'kaiser':
                cfg['window_function'] = np.kaiser
            else:
                cfg['window_function'] = np.hamming

    except Exception as e:
        print('Error:', e, '// using default config')
        cfg = {
            'window_length': 0.025,
            'window_step': 0.01,
            'cepstrum_number': 13,
            'filter_number': 26,
            'preemphasis_filter': 0.97,
            'window_function': 'hamming',
            'delta_sample': 2,
            'use_delta': True,
            'delta_delta_sample': 2,
            'use_delta_delta': True
        }

    return cfg


def eachDigitGMM(data, cfg):
    """
    Compute GMM models for each digit

    :param data: matrix of MFCC parametrized vocal samples.
                    rows - following spoken numbers
                    columns - following speakers
    :param cfg: config file for GMM estimator

    :return: Dictionary of GMM models, where key is the label of a digit and value is a GMM model
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


parametrized_data = loadData('files/parametrized.p')
config = loadConfig('config/gmm.cfg')

data_ = eachDigitGMM(parametrized_data, config)

save(data_)


