import pandas as pd
import librosa
from scipy.fft import fft
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load the CSV file into a DataFrame
metadata = pd.read_csv('metadata.csv')

# Remove the 'Source File Name' column
metadata.pop('Source File Name')

# Drop rows where the 'Class Name' is not 'Fire' or 'Rain'
data_filtered = metadata[(metadata['Class Name'] == 'Fire') | (metadata['Class Name'] == 'Rain')]

# Display the first few rows of the DataFrame to understand its structure
print(metadata.head())

audios = []
# Iterate through each row in the DataFrame
for index, row in metadata.iterrows():
    # Store necessary information in the dictionary
    audio_info = {
        'class_id': row['Class ID'],
        'audio_file': './audios/' + row['Dataset File Name'],
        'sample_rate': librosa.get_samplerate('./audios/' + row['Dataset File Name']),
    }
    audios.append(audio_info)

    if(index == 2): break


x = []
y = []

# Apply Fourier transform for each audio
for index, audio_info in audios:
    audio_data, sample_rate = librosa.load(audio_info['audio_file'], sr=None)
    transformada = fft(audio_data)

    # print(len(transformada))
    x.append(np.abs(transformada)[:20000])
    y.append(audio_info['class_id'])
    
    
    # x = np.ndarray(x)

    # print(len(transformada[:10]))
    min = -1
    for i in x:
        if(len(i)<min):
            min = i
        i = [-min+index:index + min]


# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

# Train a classifier
clf = RandomForestClassifier(n_estimators=100)

clf.fit(x_train, y_train)

# Make predictions on the testing set
predictions = clf.predict(x_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)