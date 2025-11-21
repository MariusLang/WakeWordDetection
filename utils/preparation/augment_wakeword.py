import os
import random
from glob import glob

import soundfile as sf

import librosa
import numpy as np
from scipy.signal import fftconvolve

def generate_synthetic_noises(n=5, sr=16000):
    noises = []
    for i in range(n):
        dur = np.random.uniform(1.0, 3.0)  # 1–3 Sekunden
        N = int(dur * sr)
        noise = np.random.randn(N).astype(np.float32)
        noise /= (np.max(np.abs(noise)) + 1e-6)
        noises.append(noise)
    return noises


def generate_synthetic_rirs(n=5, sr=16000):
    rirs = []
    for i in range(n):
        dur = np.random.uniform(0.1, 0.5)  # 100–500 ms
        N = int(dur * sr)
        t = np.linspace(0, dur, N)
        decay = np.exp(-3 * t / dur)
        rir = np.random.randn(N) * decay
        rir /= (np.max(np.abs(rir)) + 1e-6)
        rirs.append(rir.astype(np.float32))
    return rirs


def add_noise(x, noise, snr_db):
    Nx = len(x)
    Nn = len(noise)

    # Adapt noise length
    if Nn < Nx:
        reps = int(np.ceil(Nx / Nn))
        noise = np.tile(noise, reps)
        noise = noise[:Nx]

    elif Nn > Nx:
        start = np.random.randint(0, Nn - Nx)
        noise = noise[start:start + Nx]

    assert len(noise) == len(x)

    rms_x = np.sqrt(np.mean(x ** 2))
    rms_n = np.sqrt(np.mean(noise ** 2) + 1e-9)
    snr = 10 ** (snr_db / 20)

    noise_scaled = noise * (rms_x / (rms_n * snr))

    return x + noise_scaled


def time_stretch(x, rate):
    return librosa.effects.time_stretch(x, rate=rate)


def pitch_shift(x, sr, steps):
    return librosa.effects.pitch_shift(x, sr=sr, n_steps=steps)


def random_gain(x):
    gain = np.random.uniform(low=0.6, high=1.6)
    return x * gain


def random_time_shift(x, max_shift=6000):
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(x, shift)


def reverb(x, rir):
    out = fftconvolve(x, rir)
    return out[:len(x)]


def random_augmentation(x, sr, noises, rirs):
    x_aug = x.copy()

    if random.random() < 0.7:
        noise = random.choice(noises)
        snr = random.uniform(0, 25)
        x_aug = add_noise(x_aug, noise, snr)

    if random.random() < 0.5:
        rate = random.uniform(0.85, 1.15)
        x_aug = time_stretch(x_aug, rate)

    if random.random() < 0.5:
        steps = random.uniform(-4, 4)
        x_aug = pitch_shift(x_aug, sr, steps)

    if random.random() < 0.7:
        x_aug = random_gain(x_aug)

    if random.random() < 0.6:
        x_aug = random_time_shift(x_aug)

    if random.random() < 0.4:
        rir = random.choice(rirs)
        x_aug = reverb(x_aug, rir)

    return x_aug


def augment_single_file(fn, out_dir, noises, rirs, sr=16000):
    x, sr = librosa.load(fn, sr=sr)

    base_name = os.path.splitext(os.path.basename(fn))[0]

    saved = []

    orig_fn = os.path.join(out_dir, f'{base_name}_orig.wav')
    sf.write(orig_fn, x, sr)
    saved.append(orig_fn)

    for i in range(200):
        x_aug = random_augmentation(x, sr, noises, rirs)
        out_fn = os.path.join(out_dir, f'{base_name}_aug{i:03d}.wav')
        sf.write(out_fn, x_aug, sr)
        saved.append(out_fn)

    return saved


def load_noises(noise_dir, sr=16000):
    noises = []
    for fn in glob(os.path.join(noise_dir, '*.wav')):
        n, _ = librosa.load(fn, sr=sr)
        noises.append(n)
    return noises


def load_rirs(rir_dir, sr=16000):
    rirs = []
    for fn in glob(os.path.join(rir_dir, '*.wav')):
        rir, _ = librosa.load(fn, sr=sr)
        rirs.append(rir)
    return rirs


if __name__ == '__main__':
    wakeword_dir = 'data/wakeword'
    out_dir = 'data/wakeword_augmented'
    noise_dir = 'data/noise'
    rir_dir = 'data/rir'

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(noise_dir, exist_ok=True)
    os.makedirs(rir_dir, exist_ok=True)

    wakeword_files = glob(os.path.join(wakeword_dir, '*.wav'))
    print(f'Found {len(wakeword_files)} wakeword files.')

    noises = generate_synthetic_noises(sr=16000)
    rirs = generate_synthetic_rirs(sr=16000)

    print(f'Generated {len(noises)} noises and {len(rirs)} rirs.')

    total = 0

    for fn in wakeword_files:
        created = augment_single_file(fn, out_dir, noises, rirs)
        total += len(created)

    print(f'Created {total} augmented files.')
