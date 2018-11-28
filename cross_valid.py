from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score
import numpy as np
import math
import pickle
from sklearn.mixture import GaussianMixture
import computing_gmm

def loadData(pathMFCC, pathGMM):
    fileMFCC = open(pathMFCC, 'rb')
    fileGMM = open(pathGMM, 'rb')
    return pickle.load(fileMFCC), pickle.load(fileGMM)


def eachDigitTest(data, config):
    aux = {}
    rr = {}
    testVal = {}
    trainVal = {}
    for key1 in data:
        for key2 in data[key1]:
            if key2 == list(data[key1].keys())[0]:
                aux = data[key1][key2]
            aux = np.concatenate((aux, data[key1][key2]), axis=0)
        rr1, train1, test1 = calcRR(aux, config)
        #tutaj trzeba zapis zrobić

    return rr




def calcRR(mfcc_matrix, cfg):

    est = GaussianMixture(n_components=cfg['components'], covariance_type=cfg['covariance_type'], max_iter=cfg['max_iterations'], tol=cfg['toleration'])
    kf = KFold(n_splits=11, random_state=None)

    rr = {}
    train = {}
    test = {}

    for k, (train, test) in enumerate(kf.split(X=mfcc_matrix)):

        rr = np.round(np.exp(est.fit(mfcc_matrix[train]).score_samples(mfcc_matrix[test])), 4)
        rr = np.mean(rr)

        print(rr)

    return rr, train, test

config = computing_gmm.loadConfig('config/gmm.cfg')
MFCC, GMM = loadData('files/parametrized.p', 'files/digits_gmm.p')

data = eachDigitTest(MFCC, config)
