from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
import os
import pickle


def audio_reader(path):
    sampleRate, channel = wav.read(path)
    channel = channel/32768
    return channel, sampleRate


def loadConfig(path):
    try:
        file = open(path, 'r')
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
            'window_function': 'hamming'
        }

    return cfg


def computeMFCC(data, fs, cfg):
    fft_size = 2
    while fft_size < cfg['window_length'] * fs:
        fft_size *= 2

    data_mfcc = mfcc(data, samplerate=fs, nfft=fft_size, winlen=cfg['window_length'], winstep=cfg['window_step'],
                     numcep=cfg['cepstrum_number'], nfilt=cfg['filter_number'], preemph=cfg['preemphasis_filter'],
                     winfunc=cfg['window_function'])

    return data_mfcc


def getData(directory, file):
    path = directory + '/' + file
    samples, rate = audio_reader(path)
    return samples, rate, file[6] + '_' + file[:5], path


def save(obj):
    file = open('files/parametrized.p', 'wb')
    pickle.dump(obj, file)


config = loadConfig('config/mfcc.cfg')

#print('config:')
#for item in config:
#    print('-', item, '=', config[item])
#print('')

parameters = {}

file_directory = 'files/train'
for filename in os.listdir(file_directory):
    if filename.endswith('.wav'):
        data_ = getData(file_directory, filename)
        parameters[data_[2]] = (computeMFCC(data_[0], data_[1], config), data_[3])
        #print(parameters[data_[2]])

save(parameters)
