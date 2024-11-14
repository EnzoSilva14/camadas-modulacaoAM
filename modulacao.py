import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import sys
from suaBibSignal import signalMeu
import soundfile as sf
from scipy import signal
from scipy.signal import butter, lfilter
from math import *


def modulacaoAM(dados, fs):
    f_c = 14000
    A_c = 1
    A_m = 1

    duration = 6

    lista_tempo = np.linspace(0, duration, len(dados), endpoint=False)
    
    C = A_c * np.cos(2 * np.pi * f_c * lista_tempo)
    
    S = (dados) * (C)
    
    return S

def filtro_passa_baixa(signal, cutoff, fs, order=10):
    
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    
    # Obtenha os coeficientes do filtro Butterworth
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Aplique o filtro ao sinal
    return lfilter(b, a, signal)

def main():
    signal2 = signalMeu()
    data, fs = sf.read('gravacao.wav')
    duration = 6

    # Gerar Gráfico 1: Sinal de áudio original normalizado – domínio do tempo
    plt.figure()
    plt.plot(data)
    plt.title("Gráfico 1: Sinal de Áudio Original - Domínio do Tempo")
    plt.xlabel("Amostras")
    plt.ylabel("Amplitude")

    # ----------------- Aplicando filtro passa-baixa Butterworth ------------------

    # Aplicar o filtro Butterworth ao áudio
    yFiltrado = filtro_passa_baixa(data, 3000, fs)
    # Gráfico 2: Sinal de áudio filtrado – domínio do tempo
    plt.figure()
    plt.plot(yFiltrado)
    plt.title("Gráfico 2: Sinal de Áudio Filtrado – Domínio do Tempo")
    plt.xlabel("Amostras")
    plt.ylabel("Amplitude")
    # Gerar o vetor de tempo
    # t = np.linspace(0, duration, len(dados), endpoint=False)

    # ------- Normalizando o sinal filtrado ----------------

    max_value = np.abs(yFiltrado).max()
    audioNormalizado = yFiltrado / max_value
    
    sf.write('gravacao_filtrada.wav', audioNormalizado, fs)
    print("Arquivo filtrado salvo com sucesso!")
    sd.wait()
    sd.play(yFiltrado, samplerate=fs)
    sd.wait()
    # ------------------------------ Modulação ----------------------------
    data_filtrado, fs_filtrado = sf.read('gravacao_filtrada.wav')
    resultado_modulado = modulacaoAM(data_filtrado, fs_filtrado)
    # Gráfico 4: Sinal de áudio modulado – domínio do tempo
    plt.figure()
    plt.plot(resultado_modulado)
    plt.title("Gráfico 4: Sinal de Áudio Modulado – Domínio do Tempo")
    plt.xlabel("Amostras")
    plt.ylabel("Amplitude")
    # Normalizando o sinal modulado
    max_resultado = np.abs(resultado_modulado).max()
    audioModulado = resultado_modulado / max_resultado

    sf.write('gravacao_modulada.wav', audioModulado, fs_filtrado)
    print("Arquivo Modulado salvo com sucesso!")

    sd.wait()
    sd.play(resultado_modulado, samplerate=fs_filtrado)
    sd.wait()



    # Plota o FFT do sinal original
    signal2.plotFFT(data, fs)
    plt.legend(["Sinal Original"])

    # Plota o FFT do sinal filtrado - Gráfico 3
    signal2.plotFFT(yFiltrado, fs)
    plt.legend(["Sinal Filtrado"])

    # Gráfico 5: Sinal de áudio modulado – domínio da frequência (Fourier)
    signal2.plotFFT(resultado_modulado, fs_filtrado)
    plt.legend(["Sinal Modulado"])

    plt.show()


if __name__ == "__main__":
    main()


# Estou com dúvida no tópico 8