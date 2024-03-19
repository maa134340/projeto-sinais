import os
import pandas as pd
import librosa
import numpy as np
import scipy.io.wavfile

# Function to create 5ms frames from an audio file
def create_frames(audio_file, frame_length=0.5):
    # Load the audio file
    audio_data, sample_rate = librosa.load('audios/' + audio_file, sr=22050)
     
    # Calculate the number of samples in each frame
    frame_samples = int(sample_rate * frame_length)
    
    # Split the audio into frames
    frames = [audio_data[i:i+frame_samples] for i in range(0, len(audio_data), frame_samples)]
    
    return frames

# Load the CSV file
data = pd.read_csv("metadata.csv")

# Create a new directory to store the frames
frames_dir = "frames"
os.makedirs(frames_dir, exist_ok=True)

# List to store information about the new audio files
new_files_info = []

# Iterate through each audio file
for index, row in data.iterrows():
    source_file = row['Source File Name']
    dataset_file = row['Dataset File Name']
    class_id = row['Class ID']
    class_name = row['Class Name']
    
    # Create frames from the audio file
    frames = create_frames(dataset_file)
    
    # Save each frame as a separate audio file
    for i, frame in enumerate(frames):
        frame_file = f"{class_name}_{index}_{i}.wav"  # Unique file name for each frame
        path = os.path.join(frames_dir, frame_file)
        # Save the frame as a WAV file
        scipy.io.wavfile.write(path, 22050, (frame * 32767).astype(np.int16))
        
        # Store information about the new audio file
        new_files_info.append({
            'Frame File Name': frame_file,
            'Class ID': class_id,
            'Class Name': class_name
        })

# Create a DataFrame from the information about the new audio files
new_files_df = pd.DataFrame(new_files_info)

path = os.path.join(frames_dir, frame_file)
# Save the information about the new audio files to a CSV file
new_files_df.to_csv("frames.csv", index=False)

print("Preprocessing completed. New audio files and CSV file generated.")
