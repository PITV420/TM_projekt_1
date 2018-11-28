from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
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


def loadData(pathMFCC, pathGMM):
    fileMFCC = open(pathMFCC, 'rb')
    fileGMM = open(pathGMM, 'rb')
    return pickle.load(fileMFCC), pickle.load(fileGMM)


def validate(data, cfg):
    estimator = GaussianMixture(n_components=cfg['components'], covariance_type=cfg['covariance_type'],
                               max_iter=cfg['max_iterations'], tol=cfg['toleration'])

    models = {}
    test_set = {}
    for i in range(10):
        train, test = train_test_split(data[i], test_size=0.2, random_state=None)
        test_set[i] = test
        train_set = train[0]
        for j in range(1, len(train)):
            train_set = np.concatenate((train_set, train[i]), axis=0)
        models[i] = estimator.fit(train_set)

    for i in test_set:
        print(models[2].score(test_set[i][0]))

    return models, test_set


config = loadConfig('config/gmm.cfg')
MFCC, GMM = loadData('files/parametrized.p', 'files/digits_gmm.p')
models, test_set = validate(MFCC, config)
