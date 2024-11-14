import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import sys
from suaBibSignal import signalMeu
import soundfile as sf



def main():
    fs = 48000  # Taxa de amostragem do meu computador
    duration = 6  # segundos de gravação
    sd.default.samplerate = fs  # taxa de amostragem padrão para gravações e reprodução de áudio

# Iniciar a gravação
    print("Gravação iniciada...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    print("Gravação finalizada!")

    sf.write('gravacao.wav', audio, fs)
    print("Arquivo salvo com sucesso!")

    sd.wait()
    sd.play(audio, samplerate=fs)
    sd.wait()
    print('terminou de tocar')


if __name__ == "__main__":
    main()

