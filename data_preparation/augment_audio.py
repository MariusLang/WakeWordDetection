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
from audiomentations import (
    Compose,
    AddBackgroundNoise,
    AddShortNoises,
    TimeStretch,
    PitchShift,
    Gain,
    Shift,
    PolarityInversion,
    AddGaussianNoise
)
from audiomentations.core.transforms_interface import BaseWaveformTransform

from pathlib import Path

# Add parent directory to path to import from utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.audio_processing import normalize_peak, normalize_rms

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

class SyntheticRirs(BaseWaveformTransform):
    def __init__(self, rirs, p=0.5):
        super().__init__(p)
        self.rirs = rirs

    def apply(self, samples: np.ndarray, sample_rate: int):
        rir = random.choice(self.rirs)
        out = fftconvolve(samples, rir, mode="full")
        return out[:len(samples)]

    def get_parameters(self):
        return {"num_rirs": len(self.rirs)}


def has_wav(dir_path: str) -> bool:
    p = Path(dir_path)
    if not p.exists() or not p.is_dir():
        return False
    return any(f.is_file() and f.suffix.lower() == ".wav" for f in p.rglob("*"))

def create_augmentation_pipeline(args, rirs):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    noises_dir = os.path.join(base_dir, "noises")
    noises_background_dir = os.path.join(noises_dir, "background")
    short_noises_dir = os.path.join(noises_dir, "short")

    sr = args.sr if args.sr else 16000
    transforms = []
    if args.synthetic_noise:
        transforms.append(
            AddGaussianNoise(
                min_amplitude=0.001,
                max_amplitude=0.015,
                p=args.synthetic_noise_p
            )
        )
    if args.natural_noise and has_wav(noises_background_dir): #Background natural noise
        transforms.append(
            AddBackgroundNoise(sounds_path=noises_background_dir,
                min_snr_db=0, max_snr_db=5,
                noise_transform=PolarityInversion(),
                p=args.natural_noise_p
            )
        )

    if args.natural_noise and has_wav(short_noises_dir): #Short natural noise
        transforms.append(
            AddShortNoises(sounds_path=short_noises_dir,
                min_snr_db=0, max_snr_db=30,
                noise_rms="relative",
                min_time_between_sounds=0.25, max_time_between_sounds=1.0,
                noise_transform=PolarityInversion(),
                p=args.natural_noise_p
            )
        )

    if args.time_stretch:                                  #Time stretch
        transforms.append(
            TimeStretch(
                min_rate=0.85, max_rate=1.15,
                p=args.time_stretch_p
            )
        )

    if args.pitch_shift:                                   #Pitch shift
        transforms.append(
            PitchShift(
                min_semitones=-4, max_semitones=4,
                p=args.pitch_shift_p
            )
        )

    if args.gain:                                           #Random gain
        transforms.append(
            Gain(
                min_gain_db=-10, max_gain_db=6,
                p=args.gain_p
            )
        )

    if args.shift:                                          #Time shift
        transforms.append(
            Shift(
                min_shift=-0.25, max_shift=0.25,
                p=args.shift_p
            )
        )

    if args.reverb:                                         #Synthetic reverb
        transforms.append(
            SyntheticRirs(rirs=rirs, p=args.reverb_p)
        )

    return Compose(transforms)


def augment_single_file(fn, out_dir, augmentation, input_dir, num_augmentations=200, sr=16000,
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
        x_aug = augmentation(samples=x, sample_rate=sr)
        # Clip to prevent values outside [-1, 1] range
        x_aug = np.clip(x_aug, -1.0, 1.0)
        if len(x_aug) != len(x):
            if len(x_aug) > len(x):
                x_aug = x_aug[:len(x)]
            else:
                x_aug = np.pad(x_aug, (0, len(x) - len(x_aug)))
        out_fn = os.path.join(out_dir, f'{base_name}_aug{i:03d}.wav')
        sf.write(out_fn, x_aug, sr)
        saved.append(out_fn)

    return saved


def augment_file_worker(fn, out_dir, input_dir, rirs, num_augmentations, sr,
                        normalize_input, normalize_method, target_level, args):
    augmentation = create_augmentation_pipeline(args, rirs)
    try:
        created = augment_single_file(fn, out_dir, augmentation, input_dir, num_augmentations, sr,
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

def write_augmentation_report(out_dir, input_dir, augment, file_details, rirs, start_time, end_time,
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
        f.write(f'Synthetic RIRs generated: {len(rirs)}\n\n')

        f.write("\nAUGMENTATION PIPELINE\n")
        f.write("-" * 80 + "\n")
        for t in augment.transforms:
            f.write(f"- {t}\n")

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
  python augment_audio.py -i data/wakeword -o data/wakeword_augmented

  # Limit to ~1000 output files by randomly sampling input files
  python augment_audio.py -i data/wakeword -o data/wakeword_augmented --max-output-files 1000

  # Use RMS normalization instead
  python augment_audio.py -i data/wakeword -o data/wakeword_augmented --normalize-method rms --target-level 0.1

  # Disable normalization (not recommended - may cause volume bias)
  python augment_audio.py -i data/wakeword -o data/wakeword_augmented --no-normalize

  # Process with 4 workers and fewer augmentations for testing
  python augment_audio.py -i data/wakeword -o data/wakeword_augmented -j 4 --num-aug 50

  # Quick test: generate only 100 files with 10 augmentations each
  python augment_audio.py -i data/wakeword -o data/wakeword_augmented --max-output-files 100 --num-aug 10 -j 4
        '''
    )
    parser.add_argument('-i', '--input', required=True, help='Input directory containing audio files')
    parser.add_argument('-o', '--output', required=True, help='Output directory for augmented files')
    parser.add_argument('--num-aug', type=int, default=200, help='Number of augmentations per file (default: 200)')
    parser.add_argument('--sr', type=int, default=16000, help='Sample rate (default: 16000)')
    parser.add_argument('-j', '--jobs', type=int, default=None,
                        help='Number of parallel workers (default: number of CPU cores)')
    parser.add_argument('--normalize-method', choices=['peak', 'rms'], default='peak',
                        help='Normalization method: peak (matches training) or rms (default: peak)')
    parser.add_argument('--target-level', type=float, default=1.0,
                        help='Target level for normalization: 1.0 for peak, 0.1 for rms (default: 1.0)')
    parser.add_argument('--no-normalize', action='store_true',
                        help='Disable input normalization (NOT RECOMMENDED - may cause volume bias)')
    parser.add_argument('--max-output-files', type=int, default=None,
                        help='Maximum number of output files to generate. If specified, randomly samples input files uniformly.')

    parser.add_argument("--synthetic-noise", action="store_true", help="Enable adding Gaussian noises")
    parser.add_argument("--synthetic-noise-p", type=float, default=0.5, help="Probability of applying Gaussian noise augmentation (default: 0.5)")
    # Augmentations
    parser.add_argument("--natural-noise", action="store_true",help="Enable adding natural background and short noises from the noise dataset")
    parser.add_argument("--natural-noise-p", type=float, default=0.7,help="Probability of applying natural noise augmentation (default: 0.7)")

    parser.add_argument("--time-stretch", action="store_true",help="Enable time stretching augmentation")
    parser.add_argument("--time-stretch-p", type=float, default=0.5,help="Probability of applying time stretching (default: 0.5)")

    parser.add_argument("--pitch-shift", action="store_true",help="Enable pitch shifting augmentation")
    parser.add_argument("--pitch-shift-p", type=float, default=0.5,help="Probability of applying pitch shifting (default: 0.5)")

    parser.add_argument("--gain", action="store_true",help="Enable gain (volume scaling) augmentation")
    parser.add_argument("--gain-p", type=float, default=0.7,help="Probability of applying gain augmentation (default: 0.7)")

    parser.add_argument("--shift", action="store_true",help="Enable random time shift augmentation")
    parser.add_argument("--shift-p", type=float, default=0.6,help="Probability of applying shifting augmentation (default: 0.6)")

    parser.add_argument("--reverb", action="store_true",help="Enable synthetic reverberation using generated impulse responses")
    parser.add_argument("--reverb-p", type=float, default=0.4,help="Probability of applying reverberation augmentation (default: 0.4)")
    args = parser.parse_args()

    input_dir = args.input
    out_dir = args.output

    num_augmentations = args.num_aug
    sr = args.sr
    num_workers = args.jobs if args.jobs is not None else cpu_count()
    normalize_input = not args.no_normalize
    normalize_method = args.normalize_method
    target_level = args.target_level
    max_output_files = args.max_output_files

    os.makedirs(out_dir, exist_ok=True)

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

    rirs = generate_synthetic_rirs(sr=sr)

    print(f'Generated {len(rirs)} rirs.')
    augmentation = create_augmentation_pipeline(args, rirs)

    start_time = datetime.now()

    # Create a partial function with fixed parameters
    worker_func = partial(
        augment_file_worker,
        out_dir=out_dir,
        input_dir=input_dir,
        rirs=rirs,
        num_augmentations=num_augmentations,
        sr=sr,
        normalize_input=normalize_input,
        normalize_method=normalize_method,
        target_level=target_level,
        args=args
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
        out_dir, input_dir, augmentation, file_details, rirs,
        start_time, end_time, num_augmentations, num_workers,
        normalize_input, normalize_method, target_level,
        total_files_found, sampled
    )
    print(f'Documentation saved to: {doc_path}')
