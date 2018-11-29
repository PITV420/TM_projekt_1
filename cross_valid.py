from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

import matplotlib.pyplot as plt
from scipy.stats import norm


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


def getSets(data):
    train_idx, test_idx = train_test_split(range(len(data[0])), test_size=0.2, random_state=None)
    train_set = []
    test_set = []
    for i in range(10):
        train = []
        test = []
        for j in train_idx:
            train.append(data[i][j])
        for j in test_idx:
            test.append(data[i][j])
        train_set.append(train)
        test_set.append(test)

    return train_set, test_set


def analyze(mfcc, gmm, digit):
    """HELPER FUNCTION"""
    m = 1
    plt.hist(mfcc[:, m], bins=70, density=True, rwidth=0.90)

    left, right = plt.xlim()
    x = np.linspace(left, right, 100)
    gauss = 0
    for i in range(len(gmm.weights_)):
        y = norm.pdf(x, gmm.means_[i, m], gmm.covariances_[i, m])*gmm.weights_[i]
        plt.plot(x, y, '#EE8000')
        gauss += y
    plt.plot(x, gauss, '#DD0000')
    plt.xlim((left, right))
    plt.title('"' + str(digit) + '" - parametr c' + str(m))
    plt.show()
    print("Aic:", gmm.aic(mfcc))


def trainModels(train, cfg):
    estimator = GaussianMixture(n_components=cfg['components'], covariance_type=cfg['covariance_type'],
                               max_iter=cfg['max_iterations'], tol=cfg['toleration'])
    models = np.empty(10, dtype=GaussianMixture)
    for i in range(len(train)):
        data = train[i][0]
        for j in range(1, len(train[i])):
            data = np.concatenate((data, train[i][j]), axis=0)
        model = estimator.fit(data)
        models[i] = model
        #analyze(data, models[i], i)

    return models


def validateModels(models, test):
    matrix = np.empty((10, 10), dtype=float)
    for i in range(10):
        for j in range(10):
            matrix[i][j] = models[i].score(test[j][1])
    return matrix


config = loadConfig('config/gmm.cfg')
MFCC, MFCC_labels = loadData('files/parametrized.p', 'files/mfcc_matrix_scheme.p')

train_, test_ = getSets(MFCC)
models_ = trainModels(train_, config)
#matrix_ = validateModels(models_, test_)

"""
for i in MFCC_labels:
    print(i)

train_idx, test_idx = train_test_split(range(22), test_size=0.2, random_state=None)
print(train_idx)
print(test_idx)
train_set = []
test_set = []
for i in range(10):
    train = []
    test = []
    for j in train_idx:
        train.append(MFCC_labels[i][j])
    for j in test_idx:
        test.append(MFCC_labels[i][j])
    train_set.append(train)
    test_set.append(test)

for i in train_set:
    print(i)
for i in test_set:
    print(i)
"""
