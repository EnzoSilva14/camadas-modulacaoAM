import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from suaBibSignal import signalMeu
from scipy.signal import butter, lfilter
import soundfile as sf

def filtro_passa_baixa(signal, cutoff, fs, order=10):
    """Aplica um filtro passa-baixa Butterworth a um sinal de entrada."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, signal)

def demodulacaoAM(dados, fs, f_c=14000, cutoff=3000):
    """Realiza a demodulação AM de um sinal e aplica um filtro passa-baixa."""
    duration = len(dados) / fs
    lista_tempo = np.linspace(0, duration, len(dados), endpoint=False)
    portadora = np.cos(2 * np.pi * f_c * lista_tempo)
    
    # Multiplicação pelo sinal da portadora para demodulação
    sinal_misturado = dados * portadora
    
    # Aplicação do filtro passa-baixa
    sinal_demodulado = filtro_passa_baixa(sinal_misturado, cutoff, fs)
    
    return sinal_demodulado, sinal_misturado

def main():
    print("Ouvindo arquivo modulado")
    # Leitura do arquivo de áudio modulado
    data, fs = sf.read('gravacao_modulada.wav')
    
    # Reprodução do sinal modulado
    sd.play(data, samplerate=fs)
    sd.wait()

    # Demodulação do sinal AM
    sinal_demodulado, sinal_misturado = demodulacaoAM(data, fs)
    
    # Normalização do sinal demodulado
    max_value = np.abs(sinal_demodulado).max()
    audio_normalizado = sinal_demodulado / max_value if max_value != 0 else sinal_demodulado
    
    # Salvamento do áudio demodulado
    sf.write('gravacao_demodulada.wav', audio_normalizado, fs)
    print("Arquivo demodulado salvo com sucesso!")
    
    # Criação dos gráficos
    
    # Gráfico 6: Sinal de áudio demodulado no domínio do tempo
    plt.plot(sinal_demodulado)
    plt.title("Sinal de Áudio Demodulado - Domínio do Tempo")
    plt.xlabel("Amostras")
    plt.ylabel("Amplitude")
    
    # Instanciação do objeto de sinal para plotagem de FFT
    signal2 = signalMeu()

    # Gráfico 7: Sinal demodulado no domínio da frequência
    signal2.plotFFT(sinal_misturado, fs)
    plt.title("Sinal de Áudio Modulado - Domínio da Frequência")
    
    # Gráfico 8: Sinal demodulado e filtrado no domínio da frequência
    signal2.plotFFT(sinal_demodulado, fs)
    plt.title("Sinal de Áudio Demodulado e Filtrado - Domínio da Frequência")
    plt.show()

    # Reprodução do sinal demodulado
    print("Ouvindo arquivo demodulado")
    sd.play(audio_normalizado, samplerate=fs)
    sd.wait()


if __name__ == "__main__":
    main()
