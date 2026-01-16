import os
import json
import argparse
import numpy as np
import torch

from utils.data_loader import load_config, compute_mel_spectrogram
from utils.audio_processing import normalize_segments
from model.model_registry import get_model
from utils.get_device import get_device


def preprocess_audio(fn: str, cfg: dict):
    """
    Preprocess audio file into mel spectrogram segments.
    """
    sr = cfg['SR']
    n_mels = cfg['N_MELS']
    n_fft = cfg['N_FFT']
    hop = cfg['HOP']
    segment_frames = cfg['SEGMENT_FRAMES']

    spec = compute_mel_spectrogram(fn, sr, n_fft, hop, n_mels)

    if spec.shape[1] < segment_frames:
        pad = segment_frames - spec.shape[1]
        spec = np.pad(spec, ((0, 0), (0, pad)), mode='constant')

    segments = []
    for start in range(0, spec.shape[1] - segment_frames + 1, 10):
        seg = spec[:, start:start + segment_frames]
        segments.append(seg)

    segments = np.array(segments)
    segments_norm = normalize_segments(segments)
    segments_norm = np.expand_dims(segments_norm, axis=1)

    return torch.tensor(segments_norm, dtype=torch.float32)


def infer_model_arch_from_path(model_path: str) -> str:
    path_lower = model_path.lower()
    if 'crnn_temporal' in path_lower:
        return 'crnn_temporal'
    if 'crnn' in path_lower:
        return 'crnn'
    return 'cnn'


def load_model_from_path(model_path: str, device: torch.device):
    cfg = load_config()
    input_shape = (1, cfg['N_MELS'], cfg['SEGMENT_FRAMES'])
    num_classes = len(cfg['CLASSES'])

    if os.path.isdir(model_path):
        config_path = os.path.join(model_path, 'config.json')
        pt_path = os.path.join(model_path, 'model.pt')

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                exp_config = json.load(f)
                model_arch = exp_config.get('model_architecture', 'cnn')
                print(f'Detected model architecture: {model_arch}')
        else:
            print('Warning: config.json not found, defaulting to CNN')
            model_arch = 'cnn'
    else:
        pt_path = model_path
        parent_dir = os.path.dirname(model_path)
        config_path = os.path.join(parent_dir, 'config.json')

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                exp_config = json.load(f)
                model_arch = exp_config.get('model_architecture', 'cnn')
                print(f'Detected model architecture from config.json: {model_arch}')
        else:
            model_arch = infer_model_arch_from_path(model_path)
            print(f'Inferred model architecture from filename: {model_arch}')

    model = get_model(model_arch, input_shape, num_classes).to(device)
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.eval()

    print(f'Loaded model from: {pt_path}')
    return model


def predict_wakeword(audio_file: str, model_path: str, threshold: float = 0.2) -> bool:
    """
    Predict if audio file contains wake word.

    Args:
        audio_file: Path to audio WAV file
        model_path: Path to model .pt file or experiment directory
        threshold: Detection threshold (default: 0.2)

    Returns:
        bool: True if wake word detected
    """
    cfg = load_config()
    device = get_device()
    print(f'Using device: {device}')

    model = load_model_from_path(model_path, device)

    X = preprocess_audio(audio_file, cfg).to(device)
    print(f'Created {X.shape[0]} segments from file.')

    with torch.no_grad():
        out = model(X)
        probs = torch.softmax(out, dim=1)

    probs_np = probs.cpu().numpy()
    wake_probs = probs_np[:, 1]

    print('\n--- WakeWord Detection ---')
    print(f'Max wakeword prob: {float(np.max(wake_probs)):.4f}')

    preds = np.argmax(probs_np, axis=1)
    n_wake = np.sum(preds == 1)

    print(f'Frames predicted as wakeword: {n_wake}/{len(preds)}')

    ratio = n_wake / len(preds)
    detected = ratio > threshold

    if detected:
        print('WAKEWORD DETECTED')
    else:
        print('No wakeword')

    return detected


def main():
    parser = argparse.ArgumentParser(
        description='Run inference on audio file with trained PyTorch model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('audio_file', type=str, help='Path to audio WAV file')
    parser.add_argument('--model', type=str, default='wakeword_cnn.pt',
                        help='Path to model .pt file or experiment directory (default: wakeword_cnn.pt)')
    parser.add_argument('--threshold', type=float, default=0.2,
                        help='Detection threshold (default: 0.2)')

    args = parser.parse_args()

    predict_wakeword(args.audio_file, args.model, args.threshold)


if __name__ == '__main__':
    main()
