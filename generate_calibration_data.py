import os
import numpy as np
import glob
import argparse
from tqdm import tqdm

from utils.data_loader import load_config
from utils.audio_processing import compute_mel_spectrogram, normalize_segments


def load_audio_files(data_dir, classes, samples_per_class=None):
    """
    Load audio file paths from the training dataset.

    Args:
        data_dir: Base directory containing class subdirectories
        classes: List of class names
        samples_per_class: Dict mapping class_idx to number of samples, or int for equal samples

    Returns:
        file_paths, labels
    """
    file_paths = []
    labels = []

    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        wav_files = sorted(glob.glob(os.path.join(class_dir, '**/*.wav'), recursive=True))

        # Shuffle to get random samples
        np.random.shuffle(wav_files)

        # Determine number of samples for this class
        if samples_per_class is None:
            max_samples = len(wav_files)
        elif isinstance(samples_per_class, dict):
            max_samples = samples_per_class.get(class_idx, len(wav_files))
        else:
            max_samples = samples_per_class

        # Limit samples
        wav_files = wav_files[:min(max_samples, len(wav_files))]

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
                                 balance_classes=True,
                                 wakeword_ratio=None,
                                 min_samples_per_class=5):
    """
    Generate calibration dataset from training data.

    Args:
        output_dir: Directory to save .npy calibration files
        num_samples: Total number of calibration samples to generate
        balance_classes: If True, generate equal samples from each class (ignored if wakeword_ratio is set)
        wakeword_ratio: Ratio of wakeword samples (0.0 to 1.0). E.g., 0.2 = 20% wakeword, 80% non-wakeword
                       If set, this overrides balance_classes
        min_samples_per_class: Minimum samples per class to ensure adequate representation
    """
    # Load configuration
    cfg = load_config()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    DATA_DIR = cfg['DATA_DIR']
    CLASSES = cfg['CLASSES']

    print(f"Loading audio files from {DATA_DIR}...")
    print(f"Classes: {CLASSES}")

    # Calculate samples per class
    if wakeword_ratio is not None:
        # Use custom ratio (assumes 2 classes: non-wakeword=0, wakeword=1)
        if len(CLASSES) != 2:
            raise ValueError("wakeword_ratio only works with binary classification (2 classes)")

        n_wakeword = int(num_samples * wakeword_ratio)
        n_non_wakeword = num_samples - n_wakeword

        # Ensure minimum samples per class
        n_wakeword = max(n_wakeword, min_samples_per_class)
        n_non_wakeword = max(n_non_wakeword, min_samples_per_class)

        # Adjust total if minimums were applied
        actual_total = n_wakeword + n_non_wakeword
        actual_ratio = n_wakeword / actual_total if actual_total > 0 else 0

        samples_per_class = {0: n_non_wakeword, 1: n_wakeword}

        print(f"\nUsing imbalanced distribution:")
        print(f"  Wakeword ratio: {wakeword_ratio:.2%} (requested) -> {actual_ratio:.2%} (actual)")
        print(f"  Non-wakeword: {n_non_wakeword} samples ({n_non_wakeword / actual_total:.1%})")
        print(f"  Wakeword: {n_wakeword} samples ({n_wakeword / actual_total:.1%})")
        print(f"  Total: {actual_total} samples")

        file_paths, labels = load_audio_files(DATA_DIR, CLASSES, samples_per_class=samples_per_class)

    elif balance_classes:
        samples_per_class = num_samples // len(CLASSES)
        print(f"\nUsing balanced distribution: {samples_per_class} samples per class")
        file_paths, labels = load_audio_files(DATA_DIR, CLASSES, samples_per_class=samples_per_class)
    else:
        file_paths, labels = load_audio_files(DATA_DIR, CLASSES)
        # Randomly select num_samples files
        indices = np.random.choice(len(file_paths), min(num_samples, len(file_paths)), replace=False)
        file_paths = [file_paths[i] for i in indices]
        labels = [labels[i] for i in indices]

    print(f"\nGenerating {len(file_paths)} calibration samples...")
    class_counts = {CLASSES[i]: labels.count(i) for i in range(len(CLASSES))}
    print(f"Final class distribution: {class_counts}")

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
    parser = argparse.ArgumentParser(
        description='Generate calibration data for Hailo optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Balanced 50/50 distribution (default)
  %(prog)s --num-samples 50

  # Imbalanced: 20%% wakeword, 80%% non-wakeword (matches production)
  %(prog)s --num-samples 50 --wakeword-ratio 0.2

  # Imbalanced: 30%% wakeword, 70%% non-wakeword
  %(prog)s --num-samples 100 --wakeword-ratio 0.3

  # Random selection without balancing
  %(prog)s --num-samples 50 --no-balance
        """
    )
    parser.add_argument('--output-dir', type=str, default='calibration_data',
                        help='Output directory for calibration files (default: calibration_data)')
    parser.add_argument('--num-samples', type=int, default=50,
                        help='Total number of calibration samples to generate (default: 50)')
    parser.add_argument('--wakeword-ratio', type=float, default=None,
                        help='Ratio of wakeword samples (0.0-1.0). E.g., 0.2 = 20%% wakeword, 80%% non-wakeword. '
                             'Use this to match your production data distribution.')
    parser.add_argument('--min-per-class', type=int, default=5,
                        help='Minimum samples per class (default: 5)')
    parser.add_argument('--no-balance', action='store_true',
                        help='Do not balance classes (random selection, ignores --wakeword-ratio)')

    args = parser.parse_args()

    # Validate wakeword ratio
    if args.wakeword_ratio is not None:
        if not 0.0 <= args.wakeword_ratio <= 1.0:
            parser.error("--wakeword-ratio must be between 0.0 and 1.0")
        if args.no_balance:
            print("Warning: --wakeword-ratio is ignored when --no-balance is set")
            args.wakeword_ratio = None

    generate_calibration_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        balance_classes=not args.no_balance,
        wakeword_ratio=args.wakeword_ratio,
        min_samples_per_class=args.min_per_class
    )

    print("\nNext steps:")
    print(f"1. Copy the '{args.output_dir}' directory to your Hailo compilation PC")
    print("2. Run: hailo parser onnx wakeword.onnx --hw-arch hailo8l")
    print("3. Run optimization: hailo optimize wakeword.har --calib-set-path calibration_data/ --hw-arch hailo8l")
    print("4. Compile: hailo compiler wakeword_optimized.har --hw-arch hailo8l --batch-size 1")


if __name__ == '__main__':
    main()
