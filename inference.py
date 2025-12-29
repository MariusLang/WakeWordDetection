import sys
import torch
import numpy as np
import json
import os
import argparse

from utils.data_loader import load_config, compute_mel_spectrogram
from utils.audio_processing import normalize_segments
from model.model_registry import get_model, list_models
from utils.get_device import get_device


def preprocess_audio(fn, cfg):
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

    segments_norm = np.expand_dims(segments_norm, axis=1)

    return torch.tensor(segments_norm, dtype=torch.float32)


def load_model_from_path(model_path, device):
    cfg = load_config()
    input_shape = (1, cfg['N_MELS'], cfg['SEGMENT_FRAMES'])
    num_classes = len(cfg['CLASSES'])

    # If model_path is a directory, look for model.pt and config.json
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
        # Direct .pt file - try to infer from filename
        pt_path = model_path
        if 'crnn' in model_path.lower():
            model_arch = 'crnn'
        else:
            model_arch = 'cnn'
        print(f'Inferred model architecture from filename: {model_arch}')

    # Create and load model
    model = get_model(model_arch, input_shape, num_classes).to(device)
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.eval()

    print(f'Loaded model from: {pt_path}')
    return model


def predict_wakeword(fn, model_path):
    cfg = load_config()

    device = get_device()
    print(f'Using device {device}')

    model = load_model_from_path(model_path, device)

    X = preprocess_audio(fn, cfg).to(device)
    print(f'Created {X.shape[0]} segments from file.')

    with torch.no_grad():
        out = model(X)
        probs = torch.softmax(out, dim=1)

    probs_np = probs.cpu().numpy()
    wake_probs = probs_np[:, 1]

    print('\n--- WakeWord Detection ---')
    print('Max wakeword prob:', float(np.max(wake_probs)))

    preds = np.argmax(probs_np, axis=1)
    n_wake = np.sum(preds == 1)
    n_non = np.sum(preds == 0)

    print(f'Frames predicted as wakeword: {n_wake}/{len(preds)}')

    ratio = n_wake / len(preds)
    detected = ratio > 0.2

    if detected:
        print(' WAKEWORD DETECTED')
    else:
        print(' No wakeword')

    return detected


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run inference on audio file with trained model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default model (wakeword_cnn.pt)
  %(prog)s test.wav

  # Use specific experiment directory
  %(prog)s test.wav --model experiments/wakeword_model_20251212_163045/

  # Use specific .pt file
  %(prog)s test.wav --model my_model.pt
        """
    )

    parser.add_argument('audio_file', type=str, help='Path to audio WAV file')
    parser.add_argument('--model', type=str, default='wakeword_cnn.pt',
                        help='Path to model .pt file or experiment directory (default: wakeword_cnn.pt)')

    args = parser.parse_args()

    predict_wakeword(args.audio_file, args.model)
