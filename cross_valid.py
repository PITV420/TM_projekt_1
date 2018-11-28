from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score
import numpy as np
import math
import pickle


def loadData(pathMFCC, pathGMM):
    fileMFCC = open(pathMFCC, 'rb')
    fileGMM = open(pathGMM, 'rb')
    return pickle.load(fileMFCC), pickle.load(fileGMM)


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


def calcRR(gmm_samples, mfcc_matrix):

    kf = KFold(n_splits=22)
    rr={}

    lasso_cv = LassoCV(alphas=mfcc_matrix, cv=kf.get_n_splits(mfcc_matrix))
    for k, (train, test) in enumerate(kf.split(mfcc_matrix, gmm_samples)):
        rr = np.concatenate(rr, lasso_cv.fit(mfcc_matrix[train], gmm_samples[train]).score(mfcc_matrix[test], gmm_samples[test]))

    return rr


MFCC, GMM = loadData('files/parametrized.p', 'files/digits_gmm.p')

print(eachDigitTest(MFCC, GMM))
