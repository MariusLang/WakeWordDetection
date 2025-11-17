import matplotlib.pyplot as plt
from data_loader import compute_mel_spectrogram, load_config
import numpy as np

if __name__ == "__main__":
    cfg = load_config()
    fn = "data/bed/0bde966a_nohash_1.wav"

    spec = compute_mel_spectrogram(fn, cfg["SR"], cfg["N_FFT"], cfg["HOP"], cfg["N_MELS"])

    plt.figure(figsize=(10,4))
    plt.imshow(spec, aspect='auto', origin='lower')
    plt.title("Mel spectrogram of wakeword 'bed'")
    plt.colorbar()
    plt.show()

    print("Total frames:", spec.shape[1])
