import os
import pandas as pd
import librosa

class Preprocess:
    def __init__(self, metadata_file, audio_dir='audios', frame_length=0.025):
        self.metadata_file = metadata_file
        self.audio_dir = audio_dir
        self.frame_length = frame_length
        self.data = None
        self.labels = []

    def load_metadata(self):
        self.data = pd.read_csv(self.metadata_file)

    def split_audio(self, audio_data, class_id, frame_length=8820, hop_length=2940):

        splited_frames = librosa.util.frame(audio_data, writeable=False, frame_length=frame_length, hop_length=hop_length)

        labeled_frames = []

        for frame in splited_frames.T:
            labeled_frames.append((frame, class_id))
            
        return labeled_frames

    def create_frames(self, audio_file, class_id):
        # Load audio files
        audio_data, sample_rate = librosa.load(os.path.join(self.audio_dir, audio_file), sr= 44100)    

        # Split audio into frames
        frames = self.split_audio(audio_data, class_id)

        return frames

    def process_audio_files(self):
        self.load_metadata()
        self.frames = []

        print("Preprocessing audio files...")

        # set labels
        self.labels = self.data['Class Name'].unique()

        for _, row in self.data.iterrows():

            # extract info from csv
            dataset_file = row['Dataset File Name']
            class_id = row['Class ID']
            
            # actual preprocessing
            frames = self.create_frames(dataset_file, class_id)
            self.frames.extend(frames)

        print("Preprocessing completed. New Audio Frames created.")
