import os
import pandas as pd
import librosa

class Preprocess:
    def __init__(self, metadata_file, audio_dir='audios', frame_length=0.5):
        self.metadata_file = metadata_file
        self.audio_dir = audio_dir
        self.frame_length = frame_length
        self.data = None
        self.frames = []

    def load_metadata(self):
        self.data = pd.read_csv(self.metadata_file)

    def create_frames(self, audio_file):
        audio_data, sample_rate = librosa.load(os.path.join(self.audio_dir, audio_file), sr=22050)
        frame_samples = int(sample_rate * self.frame_length)
        frames = [audio_data[i:i+frame_samples] for i in range(0, len(audio_data), frame_samples)]
        return frames

    def process_audio_files(self):
        self.load_metadata()
        self.frames = []

        for index, row in self.data.iterrows():
            dataset_file = row['Dataset File Name']
            frames = self.create_frames(dataset_file)
            self.frames.extend(frames)

        print("Preprocessing completed. Frames created.")
