# bibliotecas para processamento do sinal
from preprocessing import Preprocess
from feature_extraction import FeatureExtractor

# bibliotecas para classificacap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# pre-processar os arquivos de audio
preprocess = Preprocess(metadata_file="metadata.csv")
preprocess.process_audio_files()

# extrair caracteristicas
feature_extractor = FeatureExtractor()
x, y = feature_extractor.extract_features(preprocess.frames)

# Split the dataset into training and testing sets
print("Splitting dataset into training and testing sets...")
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

# treinar classificador
clf = RandomForestClassifier()

# treinar classificador
print("Training classifier...")
clf.fit(x_train, y_train)

# gerar previsoes
print("Predicting...")
predictions = clf.predict(x_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)