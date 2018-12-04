import pickle
import csv
import os
import numpy as np
import operator
from python_speech_features import delta
from parametrization import loadConfig, computeMFCC, audio_reader


def loadData(dir, cfg):
    """ Load and parametrize data for scoring samples """

    """
    takes:
        dir - string containing path to a directory with samples to score
        cfg - config file for MFCC parametrization
        
    returns:
        dictionary of parametrized samples, whose key is file name and value is MFCC matrix
    """

    data_par = {}
    for filename in os.listdir(dir):
        if filename.endswith('.wav'):
            path = dir + '/' + filename
            data_raw, rate = audio_reader(path)

            data_mfcc = computeMFCC(data_raw, rate, cfg)
            data_par[filename] = data_mfcc

            data_delta = delta(data_mfcc, cfg['delta_sample'])
            data_par[filename] = np.concatenate((data_par[filename], data_delta), axis=1)

            data_delta_delta = delta(data_mfcc, cfg['delta_delta_sample'])
            data_par[filename] = np.concatenate((data_par[filename], data_delta_delta), axis=1)

    return data_par


def loadModels(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def scoreSamples(data, models):
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


config = loadConfig('config/mfcc.cfg')
mfcc_data = loadData('files/eval', config)
gmm_models = loadModels('files/digits_gmm.p')
scores_ = scoreSamples(mfcc_data, gmm_models)
saveResult(scores_)
