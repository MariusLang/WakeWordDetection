import os
import sys
import random
import argparse
from glob import glob
from datetime import datetime
import platform
from multiprocessing import Pool, cpu_count
from functools import partial

import soundfile as sf
from tqdm import tqdm

import librosa
import numpy as np
from scipy.signal import fftconvolve
from audiomentations import AddBackgroundNoise, PolarityInversion, AddShortNoises
from pathlib import Path

# Add parent directory to path to import from utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.audio_processing import normalize_peak, normalize_rms


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

def has_wav(dir_path: str) -> bool:
    p = Path(dir_path)
    if not p.exists() or not p.is_dir():
        return False
    return any(f.is_file() and f.suffix.lower() == ".wav" for f in p.rglob("*"))


def add_natural_noise(x, sr=16000):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    noises_dir = os.path.join(base_dir, "noises")

    noises_background_dir = os.path.join(noises_dir, "background")
    noised_x = x.copy()

    if has_wav(noises_background_dir):
        background_transform = AddBackgroundNoise(
            sounds_path=noises_background_dir, #load random noise from folder WakeWordDetection/data_preparation/noises
            min_snr_db=0.0,
            max_snr_db=10.0,
            noise_transform=PolarityInversion(),
            p=1.0
        )
        noised_x = background_transform(x, sample_rate=sr)

    short_noises_dir = os.path.join(noises_dir, "short")
    if has_wav(short_noises_dir):
        short_transform = AddShortNoises(
            sounds_path=short_noises_dir,
            min_snr_db=0.0,
            max_snr_db=30.0,
            noise_rms="relative",
            min_time_between_sounds=0.25,
            max_time_between_sounds=1.0,
            noise_transform=PolarityInversion(),
            p=1.0
        )
        noised_x = short_transform(noised_x, sample_rate=sr)
    return noised_x


def time_stretch(x, rate):
    return librosa.effects.time_stretch(x, rate=rate)


def pitch_shift(x, sr, steps):
    return librosa.effects.pitch_shift(x, sr=sr, n_steps=steps)


def random_gain(x, min_gain=0.3, max_gain=2.0):
    gain = np.random.uniform(low=min_gain, high=max_gain)
    return x * gain


def random_time_shift(x, max_shift=6000):
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(x, shift)


def reverb(x, rir):
    out = fftconvolve(x, rir)
    return out[:len(x)]


def random_augmentation(x, sr, noises, rirs):
    x_aug = x.copy()

    #if random.random() < 0.7:
    #    noise = random.choice(noises)
    #    snr = random.uniform(0, 25)
    #    x_aug = add_noise(x_aug, noise, snr)

    if random.random() < 0.7:
        x_aug = add_natural_noise(x_aug, sr)

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


def augment_single_file(fn, out_dir, noises, rirs, input_dir, num_augmentations=200, sr=16000,
                        normalize_input=True, normalize_method='peak', target_level=1.0):
    x, sr = librosa.load(fn, sr=sr)

    # Normalize input audio to prevent model learning volume patterns
    if normalize_input:
        if normalize_method == 'rms':
            x = normalize_rms(x, target_rms=target_level)
        elif normalize_method == 'peak':
            x = normalize_peak(x, target_peak=target_level)

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

    for i in range(num_augmentations):
        x_aug = random_augmentation(x, sr, noises, rirs)
        # Clip to prevent values outside [-1, 1] range
        x_aug = np.clip(x_aug, -1.0, 1.0)
        out_fn = os.path.join(out_dir, f'{base_name}_aug{i:03d}.wav')
        sf.write(out_fn, x_aug, sr)
        saved.append(out_fn)

    return saved


def augment_file_worker(fn, out_dir, noises, rirs, input_dir, num_augmentations, sr,
                        normalize_input, normalize_method, target_level):
    try:
        created = augment_single_file(fn, out_dir, noises, rirs, input_dir, num_augmentations, sr,
                                      normalize_input, normalize_method, target_level)
        return {
            'input': fn,
            'output_count': len(created),
            'outputs': created,
            'success': True
        }
    except Exception as e:
        return {
            'input': fn,
            'output_count': 0,
            'outputs': [],
            'success': False,
            'error': str(e)
        }


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


def write_augmentation_report(out_dir, input_dir, file_details, noises, rirs, start_time, end_time,
                              num_augmentations, num_workers, normalize_input, normalize_method, target_level,
                              total_files_found=None, sampled=False):
    duration = (end_time - start_time).total_seconds()
    total = sum(detail['output_count'] for detail in file_details)
    failed = sum(1 for detail in file_details if not detail.get('success', True))

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
        if total_files_found is not None:
            f.write(f'Total files found: {total_files_found}\n')
            if sampled:
                f.write(f'Random sampling: YES (uniformly sampled {len(file_details)} files)\n')
            else:
                f.write(f'Random sampling: NO (processed all files)\n')
        f.write(f'Sample rate: 16000 Hz\n')
        f.write(f'Augmentations per file: {num_augmentations}\n')
        f.write(f'Parallel workers: {num_workers}\n')
        f.write(f'Input normalization: {normalize_input}\n')
        if normalize_input:
            f.write(f'Normalization method: {normalize_method}\n')
            f.write(f'Target level: {target_level}\n')
        f.write(f'Synthetic noises generated: {len(noises)}\n')
        f.write(f'Synthetic RIRs generated: {len(rirs)}\n\n')

        f.write('AUGMENTATION PARAMETERS:\n')
        f.write('-' * 80 + '\n')
        f.write('- Noise addition: 70% probability, SNR 0-25 dB\n')
        f.write('- Time stretching: 50% probability, rate 0.85-1.15\n')
        f.write('- Pitch shifting: 50% probability, ±4 semitones\n')
        f.write('- Random gain: 70% probability, gain 0.3-2.0 (wider range prevents volume bias)\n')
        f.write('- Time shift: 60% probability, max ±6000 samples\n')
        f.write('- Reverb: 40% probability\n\n')

        f.write('SUMMARY:\n')
        f.write('-' * 80 + '\n')
        f.write(f'Input files processed: {len(file_details)}\n')
        f.write(f'Failed files: {failed}\n')
        f.write(f'Successful files: {len(file_details) - failed}\n')
        f.write(f'Total output files created: {total}\n')
        f.write(
            f'Files per input: {total // (len(file_details) - failed) if (len(file_details) - failed) > 0 else 0} (1 original + {num_augmentations} augmented)\n\n')

        f.write('DETAILED FILE LIST:\n')
        f.write('-' * 80 + '\n')
        for detail in file_details:
            rel_input = os.path.relpath(detail['input'], input_dir)
            if detail.get('success', True):
                f.write(f'\nInput: {rel_input}\n')
                f.write(f'  Generated {detail["output_count"]} files:\n')
                f.write(f'    - 1 original copy\n')
                f.write(f'    - {detail["output_count"] - 1} augmented versions\n')
            else:
                f.write(f'\nInput: {rel_input}\n')
                f.write(f'  FAILED: {detail.get("error", "Unknown error")}\n')

        f.write('\n' + '=' * 80 + '\n')
        f.write('END OF REPORT\n')
        f.write('=' * 80 + '\n')

    return doc_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Augment audio files with noise, reverb, and other effects.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic usage with peak normalization (matches training preprocessing)
  python augment_audio.py --in data/wakeword --out data/wakeword_augmented

  # Limit to ~1000 output files by randomly sampling input files
  python augment_audio.py --in data/wakeword --out data/wakeword_augmented --max_output_files 1000

  # Use RMS normalization instead
  python augment_audio.py --in data/wakeword --out data/wakeword_augmented --normalize rms --target_level 0.1

  # Disable normalization (not recommended - may cause volume bias)
  python augment_audio.py --in data/wakeword --out data/wakeword_augmented --normalize none

  # Process with 4 workers and fewer augmentations for testing
  python augment_audio.py --in data/wakeword --out data/wakeword_augmented --workers 4 --num_augmentations 50

  # Quick test: generate only 100 files with 10 augmentations each
  python augment_audio.py --in data/wakeword --out data/wakeword_augmented --max_output_files 100 --num_augmentations 10 --workers 4
        '''
    )
    parser.add_argument('-i', '--in', dest='wakeword_in', required=True, help='Input directory containing audio files')
    parser.add_argument('-o', '--out', dest='wakeword_out', required=True, help='Output directory for augmented files')
    parser.add_argument('--noise', default='data/noise',
                        help='Directory containing noise files (default: data/noise)')
    parser.add_argument('--rir', default='data/rir', help='Directory containing RIR files (default: data/rir)')
    parser.add_argument('--num_augmentations', type=int, default=200, help='Number of augmentations per file (default: 200)')
    parser.add_argument('--sr', type=int, default=16000, help='Sample rate (default: 16000)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: number of CPU cores)')
    parser.add_argument('--normalize', choices=['peak', 'rms', 'none'], default='peak',
                        help='Normalization method: peak (matches training), rms, or none (default: peak)')
    parser.add_argument('--target_level', type=float, default=1.0,
                        help='Target level for normalization: 1.0 for peak, 0.1 for rms (default: 1.0)')
    parser.add_argument('--max_output_files', type=int, default=None,
                        help='Maximum number of output files to generate. If specified, randomly samples input files uniformly.')

    args = parser.parse_args()

    input_dir = args.wakeword_in
    out_dir = args.wakeword_out
    noise_dir = args.noise
    rir_dir = args.rir
    num_augmentations = args.num_augmentations
    sr = args.sr
    num_workers = args.workers if args.workers is not None else cpu_count()
    normalize_input = args.normalize != 'none'
    normalize_method = args.normalize if args.normalize != 'none' else 'peak'
    target_level = args.target_level
    max_output_files = args.max_output_files

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(noise_dir, exist_ok=True)
    os.makedirs(rir_dir, exist_ok=True)

    # Find all audio files
    audio_files = glob(os.path.join(input_dir, '**/*.wav'), recursive=True)
    total_files_found = len(audio_files)
    print(f'Found {total_files_found} audio files.')

    # Sample files if max_output_files is specified
    sampled = False
    if max_output_files is not None:
        files_per_input = 1 + num_augmentations  # 1 original + N augmented
        needed_input_files = int(np.ceil(max_output_files / files_per_input))

        if needed_input_files < total_files_found:
            # Randomly sample files uniformly
            random.shuffle(audio_files)
            audio_files = audio_files[:needed_input_files]
            sampled = True
            print(f'Randomly sampled {needed_input_files} files to generate ~{needed_input_files * files_per_input} output files')
        else:
            print(f'Max output files ({max_output_files}) >= total possible ({total_files_found * files_per_input}), using all files')

    print(f'Processing {len(audio_files)} input files.')
    print(f'Using {num_workers} parallel workers.')
    if normalize_input:
        print(f'Normalization: {normalize_method} (target: {target_level})')
    else:
        print('WARNING: Normalization disabled - this may cause volume bias!')

    noises = generate_synthetic_noises(sr=sr)
    rirs = generate_synthetic_rirs(sr=sr)

    print(f'Generated {len(noises)} noises and {len(rirs)} rirs.')

    start_time = datetime.now()

    # Create a partial function with fixed parameters
    worker_func = partial(
        augment_file_worker,
        out_dir=out_dir,
        noises=noises,
        rirs=rirs,
        input_dir=input_dir,
        num_augmentations=num_augmentations,
        sr=sr,
        normalize_input=normalize_input,
        normalize_method=normalize_method,
        target_level=target_level
    )

    # Process files in parallel with progress bar
    file_details = []
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(worker_func, audio_files),
            total=len(audio_files),
            desc='Processing files',
            unit='file'
        ))
        file_details.extend(results)

    end_time = datetime.now()

    # Calculate statistics
    total = sum(detail['output_count'] for detail in file_details)
    failed = sum(1 for detail in file_details if not detail.get('success', True))

    print(f'\nCreated {total} augmented files.')
    if failed > 0:
        print(f'Warning: {failed} files failed to process.')

    # Write documentation file
    doc_path = write_augmentation_report(
        out_dir, input_dir, file_details, noises, rirs,
        start_time, end_time, num_augmentations, num_workers,
        normalize_input, normalize_method, target_level,
        total_files_found, sampled
    )
    print(f'Documentation saved to: {doc_path}')
