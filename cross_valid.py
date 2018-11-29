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

    """ Get indexes of train-test subsets """
    train_index = []
    test_index = []
    for train, test in kf.split(data[0]):
        train_index.append(train)
        test_index.append(test)
    train_index = np.asarray(train_index)
    test_index = np.asarray(test_index)
    for item in train_index:
        print(item)

    """ For each split """
    #for i in range(len(train_index)):
    for i in range(1):
        """ Split data into separate digits """
        models = []
        for digit in data:
            train_set = digit[train_index[i][0]]
            for index in train_index[i]:
                if not index == train_index[i][0]:
                    train_set = np.concatenate((train_set, digit[index]), axis=0)
            estimator = GaussianMixture(n_components=cfg['components'], max_iter=cfg['max_iterations'],
                                        tol=cfg['toleration'], covariance_type=cfg['covariance_type'])
            models.append(estimator.fit(train_set))
        models = np.asarray(models)

        """ Test models for train set """
        like_matrix = []
        for model in models:
            like = []
            for digit in data:
                like.append(model.score(digit[test_index[i][0]]))
            like_matrix.append(like)

    return np.asarray(like_matrix)


config = loadConfig('config/gmm.cfg')
MFCC, MFCC_labels = loadData('files/parametrized.p', 'files/mfcc_matrix_scheme.p')
MFCC, MFCC_labels = np.asarray(MFCC), np.asarray(MFCC_labels)
matrix_ = validateDigit(MFCC, config)
