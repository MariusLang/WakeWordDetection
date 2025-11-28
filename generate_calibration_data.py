#!/usr/bin/env python3
"""
Generate calibration dataset for Hailo model optimization.

This script processes a subset of your training data and saves it in the format
required by the Hailo compiler for model optimization.
"""

import os
import numpy as np
import glob
import argparse
from tqdm import tqdm

from utils.data_loader import load_config
from utils.audio_processing import compute_mel_spectrogram, normalize_segments


def load_audio_files(data_dir, classes, max_per_class=None):
    """Load audio file paths from the training dataset."""
    file_paths = []
    labels = []

    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        wav_files = sorted(glob.glob(os.path.join(class_dir, '**/*.wav'), recursive=True))

        # Limit samples per class if specified
        if max_per_class:
            wav_files = wav_files[:max_per_class]

        for fn in wav_files:
            file_paths.append(fn)
            labels.append(class_idx)

    return file_paths, labels


def extract_segment(spec, segment_frames, label):
    """Extract a segment from the spectrogram."""
    T = spec.shape[1]

    # Pad if too short
    if T < segment_frames:
        pad = segment_frames - T
        spec = np.pad(spec, ((0, 0), (0, pad)), mode='constant')
        T = segment_frames

    if label == 1:  # Wake word - extract centered on energy peak
        energy = spec.mean(axis=0)
        peak = np.argmax(energy)

        start = peak - segment_frames // 2
        if start < 0:
            start = 0

        end = start + segment_frames
        if end > spec.shape[1]:
            end = spec.shape[1]
            start = end - segment_frames
    else:  # Non-wake word - random segment
        max_offset = T - segment_frames
        start = np.random.randint(0, max_offset + 1)

    return spec[:, start:start + segment_frames]


def preprocess_audio_for_calibration(audio_path, cfg, label):
    """
    Preprocess audio file to format expected by the model.
    Matches the exact preprocessing pipeline used during training.
    """
    SR = cfg['SR']
    N_MELS = cfg['N_MELS']
    N_FFT = cfg['N_FFT']
    HOP = cfg['HOP']
    SEGMENT_FRAMES = cfg['SEGMENT_FRAMES']

    # Compute mel spectrogram
    spec = compute_mel_spectrogram(audio_path, SR, N_FFT, HOP, N_MELS)

    # Extract segment
    segment = extract_segment(spec, SEGMENT_FRAMES, label)

    # Normalize the segment (same as training)
    segment = segment[np.newaxis, ...]  # Add batch dimension for normalize_segments
    segment_norm = normalize_segments(segment)[0]  # Remove batch dimension

    # Add channel dimension: (n_mels, segment_frames) -> (n_mels, segment_frames, 1)
    # Hailo expects HWC format (Height, Width, Channels) = (40, 100, 1)
    segment_norm = np.expand_dims(segment_norm, axis=-1)

    return segment_norm.astype(np.float32)


def generate_calibration_dataset(output_dir='calibration_data',
                                  num_samples=50,
                                  balance_classes=True):
    """
    Generate calibration dataset from training data.

    Args:
        output_dir: Directory to save .npy calibration files
        num_samples: Total number of calibration samples to generate
        balance_classes: If True, generate equal samples from each class
    """
    # Load configuration
    cfg = load_config()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    DATA_DIR = cfg['DATA_DIR']
    CLASSES = cfg['CLASSES']

    print(f"Loading audio files from {DATA_DIR}...")
    print(f"Classes: {CLASSES}")

    if balance_classes:
        samples_per_class = num_samples // len(CLASSES)
        file_paths, labels = load_audio_files(DATA_DIR, CLASSES, max_per_class=samples_per_class)
    else:
        file_paths, labels = load_audio_files(DATA_DIR, CLASSES)
        # Randomly select num_samples files
        indices = np.random.choice(len(file_paths), min(num_samples, len(file_paths)), replace=False)
        file_paths = [file_paths[i] for i in indices]
        labels = [labels[i] for i in indices]

    print(f"\nGenerating {len(file_paths)} calibration samples...")
    print(f"Class distribution: {dict(zip(CLASSES, [labels.count(i) for i in range(len(CLASSES))]))}")

    # Process each file
    for idx, (audio_path, label) in enumerate(tqdm(zip(file_paths, labels), total=len(file_paths))):
        try:
            # Preprocess audio
            features = preprocess_audio_for_calibration(audio_path, cfg, label)

            # Save as .npy file
            output_path = os.path.join(output_dir, f'sample_{idx:04d}.npy')
            np.save(output_path, features)

        except Exception as e:
            print(f"\nError processing {audio_path}: {e}")
            continue

    # Verify and print statistics
    print(f"\nCalibration dataset created successfully!")
    print(f"Location: {output_dir}")
    print(f"Total samples: {len(os.listdir(output_dir))}")

    # Load a sample to verify shape
    sample = np.load(os.path.join(output_dir, 'sample_0000.npy'))
    print(f"\nSample shape: {sample.shape}")
    print(f"Sample dtype: {sample.dtype}")
    print(f"Sample range: [{sample.min():.3f}, {sample.max():.3f}]")
    print(f"Expected shape (HWC): ({cfg['N_MELS']}, {cfg['SEGMENT_FRAMES']}, 1)")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Generate calibration data for Hailo optimization')
    parser.add_argument('--output-dir', type=str, default='calibration_data',
                        help='Output directory for calibration files (default: calibration_data)')
    parser.add_argument('--num-samples', type=int, default=50,
                        help='Number of calibration samples to generate (default: 50)')
    parser.add_argument('--no-balance', action='store_true',
                        help='Do not balance classes (random selection)')

    args = parser.parse_args()

    generate_calibration_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        balance_classes=not args.no_balance
    )

    print("\nNext steps:")
    print(f"1. Copy the '{args.output_dir}' directory to your Hailo compilation PC")
    print("2. Run: hailo parser onnx wakeword.onnx --hw-arch hailo8l")
    print("3. Run optimization: hailo optimize wakeword.har --calib-set-path calibration_data/ --hw-arch hailo8l")
    print("4. Compile: hailo compiler wakeword_optimized.har --hw-arch hailo8l --batch-size 1")


if __name__ == '__main__':
    main()
