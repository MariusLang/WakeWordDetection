import sys
import torch
import numpy as np

from utils.data_loader import load_config, compute_mel_spectrogram
from train_cnn import WakeWordCNN, normalize_segments
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


def predict_wakeword(fn):
    cfg = load_config()

    device = get_device()
    print(f'Using device {device}')

    input_shape = (1, cfg['N_MELS'], cfg['SEGMENT_FRAMES'])
    num_classes = len(cfg['CLASSES'])
    model = WakeWordCNN(input_shape, num_classes).to(device)
    model.load_state_dict(torch.load('wakeword_cnn.pt', map_location=device))
    model.eval()

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
    if len(sys.argv) < 2:
        print('Usage:\n  python inference.py path/to/audio.wav')
        sys.exit(1)

    fn = sys.argv[1]
    predict_wakeword(fn)
