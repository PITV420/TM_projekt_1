from sklearn.model_selection import cross_val_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LassoCV
import numpy as np
import math
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


"""
def eachDigitTest(data, GMM):
    aux = {}
    rr = {}
    for key1 in data:
        for key2 in data[key1]:
            if key2 == list(data[key1].keys())[0]:
                aux = data[key1][key2]
            aux = np.concatenate((aux, data[key1][key2]), axis=0)
        rr[key1] = calcRR(GMM[key1].score_samples(aux), aux)
    return rr
"""

"""
def calcRR(gmm_samples, mfcc_matrix):

    kf = KFold(n_splits=22)
    rr={}

    lasso_cv = LassoCV(alphas=mfcc_matrix, cv=kf.get_n_splits(mfcc_matrix))
    for k, (train, test) in enumerate(kf.split(mfcc_matrix, gmm_samples)):
        rr = np.concatenate(rr, lasso_cv.fit(mfcc_matrix[train], gmm_samples[train]).score(mfcc_matrix[test], gmm_samples[test]))

    return rr
"""


def validate(data, cfg):
    estimator = GaussianMixture(n_components=cfg['components'], covariance_type=cfg['covariance_type'],
                               max_iter=cfg['max_iterations'], tol=cfg['toleration'])

    train, test = train_test_split(data[0], test_size=0.2, random_state=None)
    #cv = cross_val_score(estimator, data['0'], cv=5)

    return train, test


config = loadConfig('config/gmm.cfg')
MFCC, GMM = loadData('files/parametrized.p', 'files/digits_gmm.p')
train, test = validate(MFCC, config)
#print(eachDigitTest(MFCC, GMM))
