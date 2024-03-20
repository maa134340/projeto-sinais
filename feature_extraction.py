import numpy as np
from scipy.fft import fft
import librosa

class FeatureExtractor:
    def __init__(self):
        self.use_mfcc = False
        pass

    def extract_mfcc(self, audio_data):
        mfcc_features = librosa.feature.mfcc(y=audio_data, sr=44100)
        return mfcc_features.flatten()
    
    def extract_zero_crossing_rate(self, audio_data):
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio_data)
        return zero_crossing_rate.flatten()
    
    def extract_spectral_flatness(self, audio_data):
        flatness = librosa.feature.spectral_flatness(y=audio_data)
        return flatness.flatten()

    def extract_spectral_centroid(self, audio_data):
        centroid = librosa.feature.spectral_centroid(y=audio_data, sr = 44100)
        return centroid.flatten()    
    
    def extract_fourirer(self, audio_data):
        spectrum = np.abs(fft(audio_data))
        return spectrum
    
    def extract_bpm(self, audio_data):
        pass

    def extract_features(self, frames):

        x = []
        y = []

        print("Extracting features...")

        for frame in frames:
        
            # unpack frame
            audio_data, class_id = frame

            # get mfcc features
            mfcc_features = self.extract_mfcc(audio_data)

            # extract zero-crossing rate feature
            zero_crossing_feature = self.extract_zero_crossing_rate(audio_data)

            # extract spectral flatness feature
            spectral_flatness_feature = self.extract_spectral_flatness(audio_data)

            # extract spectral centroid feature
            spectral_centroid_feature = self.extract_spectral_centroid(audio_data)

            #extract fourirer
            fourirer = self.extract_fourirer(audio_data)

            if(self.use_mfcc):
                # concatenate MFCC and zero-crossing rate features
                combined_features = np.concatenate((mfcc_features, zero_crossing_feature, spectral_flatness_feature, spectral_centroid_feature))
            else:
                 # concatenate FFT and zero-crossing rate features
                combined_features = np.concatenate((fourirer, zero_crossing_feature, spectral_flatness_feature, spectral_centroid_feature))
            # append to x and y
            x.append(combined_features)
            y.append(class_id)

        print("Features extraction completed.")
        
        return x, y