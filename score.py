"""
Recognize samples using previously trained models and show confusion matrix
"""


import pickle
import csv
import os
import numpy as np
import operator
from python_speech_features import delta
from parametrization import loadConfig, computeMFCC, audio_reader
import evaluate
import parametrization
import computing_gmm


def loadData(dir, cfg):
    """
    Load and parametrize data for scoring samples

    :param dir: string containing path to a directory with samples to score
    :param cfg: config file for MFCC parametrization
    :return: dictionary of parametrized samples, whose key is file name and value is MFCC matrix
    """

    data_par = {}
    for filename in os.listdir(dir):
        if filename.endswith('.wav'):
            path = dir + '/' + filename
            data_raw, rate = audio_reader(path)

            data_mfcc = computeMFCC(data_raw, rate, cfg)
            data_par[filename] = data_mfcc

    return data_par


def loadModels(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def scoreSamples(data, models):
    """
    Score evaluation samples by digit models

    :param data: dictionary of evaluation data
    :param models: dictionary of trained GMM models

    :return: dictionary of labeled samples
        format:
            key,            (value1,    value2)
            file name,      label,      log-likelihood
    """

    scores = {}
    for sample in data:
        sample_scores = {}
        for digit in models:
            sample_scores[digit] = models[digit].score(data[sample])
        label = max(sample_scores.items(), key=operator.itemgetter(1))[0]
        score = sample_scores[label]
        scores[sample] = (label, score)
    return scores


def saveResult(data):
    with open('files/results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for key in data:
            writer.writerow([key, data[key][0], data[key][1]])


def main():
    """
    Parametrize evaluation set and score it according to config files

    :save: data in results.csv
    :print: confusion matrix for labeled samples
    """

    parametrization.main()
    computing_gmm.main()

    config = loadConfig('config/mfcc.cfg')
    mfcc_data = loadData('files/eval', config)
    gmm_models = loadModels('files/digits_gmm.p')
    scores_ = scoreSamples(mfcc_data, gmm_models)
    saveResult(scores_)
    evaluate.evaluate('files/results.csv')

main()