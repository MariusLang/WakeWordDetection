import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler


def normalize_peak(x, target_peak=1.0):
    """
    Normalize audio to target peak level.

    This is the standard normalization used throughout the pipeline:
    - Training data loading
    - Data augmentation
    - Inference

    Args:
        x: Audio signal array
        target_peak: Target peak amplitude (default: 1.0)

    Returns:
        Normalized audio array
    """
    current_peak = np.max(np.abs(x))
    if current_peak > 1e-6:
        return x * (target_peak / current_peak)
    return x


def normalize_rms(x, target_rms=0.1):
    """
    Normalize audio to target RMS level.

    Alternative normalization method (not used by default).

    Args:
        x: Audio signal array
        target_rms: Target RMS level (default: 0.1)

    Returns:
        Normalized audio array
    """
    current_rms = np.sqrt(np.mean(x ** 2))
    if current_rms > 1e-6:
        return x * (target_rms / current_rms)
    return x


def compute_mel_spectrogram(fn, sr, n_fft, hop, n_mels):
    """
    Compute mel spectrogram from audio file.

    Preprocessing pipeline:
    1. Load audio file
    2. Peak normalization: x / max(abs(x))  [CRITICAL: prevents volume bias]
    3. Compute mel spectrogram
    4. Convert to dB scale

    Args:
        fn: Audio file path
        sr: Sample rate
        n_fft: FFT window size
        hop: Hop length
        n_mels: Number of mel bands

    Returns:
        Mel spectrogram in dB scale
    """
    x, _ = librosa.load(fn, sr=sr)
    x = normalize_peak(x, target_peak=1.0)  # Peak normalization to [−1, 1]

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
    """
    Compute mel spectrogram from audio array (not file).
    Used for real-time inference in continuous_wakeword.py.

    Same preprocessing as compute_mel_spectrogram but from array instead of file.

    Args:
        audio_data: Audio signal array
        sr: Sample rate
        n_fft: FFT window size
        hop: Hop length
        n_mels: Number of mel bands

    Returns:
        Mel spectrogram in dB scale
    """
    x = audio_data.flatten()
    x = normalize_peak(x, target_peak=1.0)  # Peak normalization to [−1, 1]

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


def normalize_segments(X):
    """
    Normalize mel spectrogram segments using StandardScaler.
    Applied after mel spectrogram computation, before feeding to model.

    This applies z-score normalization (zero mean, unit variance) to each segment.
    Used in both training and inference.

    Args:
        X: Array of mel spectrogram segments, shape (N, n_mels, n_frames)

    Returns:
        Normalized segments with same shape
    """
    X_norm = np.zeros_like(X)
    for i in range(len(X)):
        scaler = StandardScaler()
        X_norm[i] = scaler.fit_transform(X[i])
    return X_norm


def preprocess_audio_file(fn, cfg):
    """
    Preprocess audio file into segments for inference.
    Used in rpi_wakeword.py for single file inference.

    Pipeline:
    1. Compute mel spectrogram (with peak normalization)
    2. Pad if needed
    3. Create sliding window segments
    4. Normalize segments with StandardScaler
    5. Add channel dimension for model input

    Args:
        fn: Audio file path
        cfg: Configuration dict with SR, N_MELS, N_FFT, HOP, SEGMENT_FRAMES

    Returns:
        Preprocessed segments ready for model, shape (N, n_mels, n_frames, 1)
    """
    SR = cfg['SR']
    N_MELS = cfg['N_MELS']
    N_FFT = cfg['N_FFT']
    HOP = cfg['HOP']
    SEGMENT_FRAMES = cfg['SEGMENT_FRAMES']

    spec = compute_mel_spectrogram(fn, SR, N_FFT, HOP, N_MELS)

    if spec.shape[1] < SEGMENT_FRAMES:
        pad = SEGMENT_FRAMES - spec.shape[1]
        spec = np.pad(spec, ((0, 0), (0, pad)), mode='constant')

    segments = []
    for start in range(0, spec.shape[1] - SEGMENT_FRAMES + 1, 10):
        seg = spec[:, start:start + SEGMENT_FRAMES]
        segments.append(seg)

    segments = np.array(segments)
    segments_norm = normalize_segments(segments)

    # Add channel dimension for Hailo (HWC format)
    segments_norm = np.expand_dims(segments_norm, axis=-1)

    return segments_norm.astype(np.float32)


def preprocess_audio_chunk(audio_data, cfg):
    """
    Preprocess audio chunk into segments for inference.
    Used in continuous_wakeword.py for real-time detection.

    Same as preprocess_audio_file but takes audio array instead of file path.

    Args:
        audio_data: Audio signal array
        cfg: Configuration dict with SR, N_MELS, N_FFT, HOP, SEGMENT_FRAMES

    Returns:
        Preprocessed segments ready for model, shape (N, n_mels, n_frames, 1)
        or None if no valid segments could be created
    """
    SR = cfg['SR']
    N_MELS = cfg['N_MELS']
    N_FFT = cfg['N_FFT']
    HOP = cfg['HOP']
    SEGMENT_FRAMES = cfg['SEGMENT_FRAMES']

    # Compute mel spectrogram from audio array
    spec = compute_mel_spectrogram_from_audio(audio_data, SR, N_FFT, HOP, N_MELS)

    if spec.shape[1] < SEGMENT_FRAMES:
        pad = SEGMENT_FRAMES - spec.shape[1]
        spec = np.pad(spec, ((0, 0), (0, pad)), mode='constant')

    segments = []
    for start in range(0, spec.shape[1] - SEGMENT_FRAMES + 1, 10):
        seg = spec[:, start:start + SEGMENT_FRAMES]
        segments.append(seg)

    if len(segments) == 0:
        return None

    segments = np.array(segments)
    segments_norm = normalize_segments(segments)

    # Add channel dimension for Hailo (HWC format)
    segments_norm = np.expand_dims(segments_norm, axis=-1)

    return segments_norm.astype(np.float32)
