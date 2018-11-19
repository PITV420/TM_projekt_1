from python_speech_features import mfcc
import numpy as np


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


config = loadConfig('config/mfcc.cfg')

print('config:')
for item in config:
    print('-', item, '=', config[item])
print('')
