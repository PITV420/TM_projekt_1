import scipy.io.wavfile as wav
import math
import numpy as np

def audio_reader(path):
    sampleRate, channel = wav.read(path)
    channel = channel/32768
    return sampleRate, channel