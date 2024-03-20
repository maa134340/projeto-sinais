import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.fft import fft

# Função para plotar o sinal de áudio e sua Transformada de Fourier
def plot_audio_and_fourier(file_path):
    # Carregar o arquivo de áudio com librosa
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    
    # Calcular a Transformada de Fourier
    spectrum = np.abs(fft(audio_data))
    freqs = np.fft.fftfreq(len(audio_data), 1/sample_rate)
    
    # Plotar o sinal de áudio
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(audio_data)
    plt.title('Áudio Original')
    plt.xlabel('Amostras')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # Plotar a Transformada de Fourier
    plt.subplot(2, 1, 2)
    plt.plot(freqs, spectrum)
    plt.title('Transformada de Fourier')
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Carregar o arquivo CSV
data = pd.read_csv("metadata_environmental_and_threats_class.csv")

# Função para plotar o sinal de áudio e sua Transformada de Fourier para uma determinada classe
def plot_audio_and_fourier_for_class(data, class_name):
    # Filtrar os dados pela classe fornecida
    class_data = data[data['Class Name'] == class_name]
    
    if len(class_data) == 0:
        print("Nenhum dado encontrado para a classe fornecida.")
        return
    
    # Obter o caminho do arquivo de áudio da primeira instância da classe
    file_path = 'audios/' + class_data.iloc[0]['Dataset File Name']
    
    # Plotar o sinal de áudio e sua Transformada de Fourier
    plot_audio_and_fourier(file_path)

# Verificar se o nome da classe foi fornecido como argumento da linha de comando
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <class_name>")
    else:
        class_name = sys.argv[1]
        plot_audio_and_fourier_for_class(data, class_name)