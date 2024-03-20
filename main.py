# Libraries for signal processing
from preprocessing import Preprocess
from feature_extraction import FeatureExtractor

# Libraries for classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Libraries for evaluation
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Preprocess audio files
preprocess = Preprocess(metadata_file="metadata_environmental_class.csv")
preprocess.process_audio_files()

# Extract features
feature_extractor = FeatureExtractor()
x, y = feature_extractor.extract_features(preprocess.frames)

# Split the dataset into training and testing sets
print("Splitting dataset into training and testing sets...")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)

# Train classifier
clf = RandomForestClassifier()

# Train classifier
print("Training classifier...")
clf.fit(x_train, y_train)

# Generate predictions
print("Predicting...")
predictions = clf.predict(x_test)

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=preprocess.labels, yticklabels=preprocess.labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Evaluate the classifier
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')

# print evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")