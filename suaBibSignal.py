
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy import signal as window

# FFT - Fast Fourier Transform

class signalMeu:
    def __init__(self):
        self.init = 0

    def __init__(self):
        self.init = 0


    def calcFFT(self, signal, fs):
        # https://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html
        N  = len(signal) #quantos pontos de amostra tem no array do sinal
        # W = window.hamming(N) #Aplica janela de hamming no sinal para suavizar reduzir os efeitos indesejados de borda (leakage) na transformada
        T  = 1/fs
        xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
        yf = fft(signal) #aplica a transformada
        return(xf, np.abs(yf[0:N//2])) #freq, magnitude

    def plotFFT(self, signal, fs):
        x,y = self.calcFFT(signal, fs)
        plt.figure()
        plt.plot(x, np.abs(y))
        plt.title('Fourier')
