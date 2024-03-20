import numpy as np
from scipy.fft import fft

class FeatureExtractor:
    def __init__(self):
        pass

    def extract_features(self, frames, metadata):
        audios = []

        for index, row in metadata.iterrows():
            class_id = row['Class ID']
            audio_data = frames[index]
            transformada = fft(audio_data)
            audios.append((np.abs(transformada)[:20000], class_id))

        x = np.array([audio[0] for audio in audios])
        y = np.array([audio[1] for audio in audios])

        return x, y
