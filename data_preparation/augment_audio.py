import os
import random
import argparse
from glob import glob
from datetime import datetime
import platform

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


def augment_single_file(fn, out_dir, noises, rirs, input_dir, sr=16000):
    x, sr = librosa.load(fn, sr=sr)

    # Preserve subdirectory structure by getting relative path
    rel_path = os.path.relpath(fn, input_dir)
    base_name = os.path.splitext(rel_path)[0]

    # Create subdirectory in output if needed
    out_subdir = os.path.dirname(os.path.join(out_dir, base_name))
    os.makedirs(out_subdir, exist_ok=True)

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


def write_augmentation_report(out_dir, input_dir, file_details, noises, rirs, start_time, end_time):
    duration = (end_time - start_time).total_seconds()
    total = sum(detail['output_count'] for detail in file_details)

    doc_path = os.path.join(out_dir, 'augmentation_report.txt')
    with open(doc_path, 'w') as f:
        f.write('=' * 80 + '\n')
        f.write('AUDIO AUGMENTATION REPORT\n')
        f.write('=' * 80 + '\n\n')

        f.write(f'Date: {start_time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Processing time: {duration:.2f} seconds\n')
        f.write(f'Device: {platform.system()} {platform.release()}\n')
        f.write(f'Machine: {platform.machine()}\n')
        f.write(f'Processor: {platform.processor()}\n')
        f.write(f'Python version: {platform.python_version()}\n\n')

        f.write('CONFIGURATION:\n')
        f.write('-' * 80 + '\n')
        f.write(f'Input directory: {input_dir}\n')
        f.write(f'Output directory: {out_dir}\n')
        f.write(f'Sample rate: 16000 Hz\n')
        f.write(f'Augmentations per file: 200\n')
        f.write(f'Synthetic noises generated: {len(noises)}\n')
        f.write(f'Synthetic RIRs generated: {len(rirs)}\n\n')

        f.write('AUGMENTATION PARAMETERS:\n')
        f.write('-' * 80 + '\n')
        f.write('- Noise addition: 70% probability, SNR 0-25 dB\n')
        f.write('- Time stretching: 50% probability, rate 0.85-1.15\n')
        f.write('- Pitch shifting: 50% probability, ±4 semitones\n')
        f.write('- Random gain: 70% probability, gain 0.6-1.6\n')
        f.write('- Time shift: 60% probability, max ±6000 samples\n')
        f.write('- Reverb: 40% probability\n\n')

        f.write('SUMMARY:\n')
        f.write('-' * 80 + '\n')
        f.write(f'Input files processed: {len(file_details)}\n')
        f.write(f'Total output files created: {total}\n')
        f.write(f'Files per input: {total // len(file_details) if file_details else 0} (1 original + 200 augmented)\n\n')

        f.write('DETAILED FILE LIST:\n')
        f.write('-' * 80 + '\n')
        for detail in file_details:
            rel_input = os.path.relpath(detail['input'], input_dir)
            f.write(f'\nInput: {rel_input}\n')
            f.write(f'  Generated {detail["output_count"]} files:\n')
            f.write(f'    - 1 original copy\n')
            f.write(f'    - {detail["output_count"] - 1} augmented versions\n')

        f.write('\n' + '=' * 80 + '\n')
        f.write('END OF REPORT\n')
        f.write('=' * 80 + '\n')

    return doc_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Augment audio files with noise, reverb, and other effects.')
    parser.add_argument('-i', '--input', required=True, help='Input directory containing audio files')
    parser.add_argument('-o', '--output', required=True, help='Output directory for augmented files')
    parser.add_argument('--noise-dir', default='data/noise', help='Directory containing noise files (default: data/noise)')
    parser.add_argument('--rir-dir', default='data/rir', help='Directory containing RIR files (default: data/rir)')
    parser.add_argument('--num-aug', type=int, default=200, help='Number of augmentations per file (default: 200)')
    parser.add_argument('--sr', type=int, default=16000, help='Sample rate (default: 16000)')

    args = parser.parse_args()

    input_dir = args.input
    out_dir = args.output
    noise_dir = args.noise_dir
    rir_dir = args.rir_dir
    num_augmentations = args.num_aug
    sr = args.sr

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(noise_dir, exist_ok=True)
    os.makedirs(rir_dir, exist_ok=True)

    audio_files = glob(os.path.join(input_dir, '**/*.wav'), recursive=True)
    print(f'Found {len(audio_files)} audio files.')

    noises = generate_synthetic_noises(sr=sr)
    rirs = generate_synthetic_rirs(sr=sr)

    print(f'Generated {len(noises)} noises and {len(rirs)} rirs.')

    total = 0
    file_details = []
    start_time = datetime.now()

    for fn in audio_files:
        created = augment_single_file(fn, out_dir, noises, rirs, input_dir, sr=sr)
        total += len(created)
        file_details.append({
            'input': fn,
            'output_count': len(created),
            'outputs': created
        })

    end_time = datetime.now()

    print(f'Created {total} augmented files.')

    # Write documentation file
    doc_path = write_augmentation_report(out_dir, input_dir, file_details, noises, rirs, start_time, end_time)
    print(f'Documentation saved to: {doc_path}')
