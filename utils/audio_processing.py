import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler


def normalize_segments(X):
    X_norm = np.zeros_like(X)
    for i in range(len(X)):
        scaler = StandardScaler()
        X_norm[i] = scaler.fit_transform(X[i])
    return X_norm


def compute_mel_spectrogram(fn, sr, n_fft, hop, n_mels):
    x, _ = librosa.load(fn, sr=sr)
    x = x / (np.max(np.abs(x)) + 1e-6)

    mel = librosa.feature.melspectrogram(
        y=x,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=n_mels,
        power=2,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


def compute_mel_spectrogram_from_audio(audio_data, sr, n_fft, hop, n_mels):
    """Compute mel spectrogram from audio array (not file)."""
    x = audio_data.flatten()
    x = x / (np.max(np.abs(x)) + 1e-6)

    mel = librosa.feature.melspectrogram(
        y=x,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=n_mels,
        power=2,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db
