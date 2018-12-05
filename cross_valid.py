"""
Cross-validate models based on data in training set
"""


from sklearn.model_selection import KFold
import numpy as np
import pickle
from sklearn.mixture import GaussianMixture

def loadConfig(path):
    """
    :param path: path to a config file
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


def loadData(pathMFCC, pathMFCC_labels):
    with open(pathMFCC, 'rb') as fileMFCC:
        dataMFCC = pickle.load(pathMFCC)

    with open(pathMFCC_labels, 'rb') as fileMFCC_labels:
        dataMFCC_labels = pickle.load(fileMFCC_labels)

    return dataMFCC, dataMFCC_labels


def validateDigit(data, cfg):
    """
    Cross-validate models

    :param data: matrix of mfcc matrices computed by parametrization.py
    :param cfg: config dictionary
    :return: confusion matrix for train-test data
    """

    kf = KFold(n_splits=5, shuffle=False, random_state=None)

    """ Get indexes of train-test subsets """
    train_index = []
    test_index = []
    for train, test in kf.split(data[0]):
        train_index.append(train)
        test_index.append(test)
    train_index = np.asarray(train_index)
    test_index = np.asarray(test_index)

    rr_matrix = np.zeros((10, 10), dtype=int)
    tests = 0
    """ For each split """
    for i in range(len(train_index)):
        """ Split data into separate digits """
        models = []
        for digit in data:
            train_set = digit[train_index[i][0]]
            for index in train_index[i]:
                if not index == train_index[i][0]:
                    train_set = np.concatenate((train_set, digit[index]), axis=0)
            estimator = GaussianMixture(n_components=cfg['components'], max_iter=cfg['max_iterations'],
                                        tol=cfg['tolerance'], covariance_type=cfg['covariance_type'])
            models.append(estimator.fit(train_set))
        models = np.asarray(models)

        """ For each speaker"""
        for j in range(len(test_index[i])):
            recog_matrix = []
            """ Check each digit """
            for digit in data:
                like = []
                recog = np.zeros(10, dtype=int)
                """ Score it by fitted models """
                for model in models:
                    like.append(model.score(digit[test_index[i][j]]))
                """ Max likelihood is recognized digit """
                recog[like.index(max(like))] = 1
                recog_matrix.append(recog)

            """ Add it to recognition matrix """
            rr_matrix = np.add(rr_matrix, recog_matrix)
            tests += 1

    return np.asarray(rr_matrix)


config = loadConfig('config/gmm.cfg')
MFCC, MFCC_labels = loadData('files/parametrized.p', 'files/mfcc_matrix_scheme.p')
MFCC, MFCC_labels = np.asarray(MFCC), np.asarray(MFCC_labels)
matrix_ = validateDigit(MFCC, config)
