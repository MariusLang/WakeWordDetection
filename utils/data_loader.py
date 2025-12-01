import os
import glob
import numpy as np
import configparser

from utils.audio_processing import compute_mel_spectrogram


def load_config(path='config.ini'):
    config = configparser.ConfigParser()
    config.read(path)

    cfg = {}

    cfg['SR'] = config.getint('audio', 'sr')
    cfg['N_MELS'] = config.getint('audio', 'n_mels')
    cfg['N_FFT'] = config.getint('audio', 'n_fft')
    cfg['HOP'] = config.getint('audio', 'hop')

    cfg['SEGMENT_FRAMES'] = config.getint('segments', 'segment_frames')
    cfg['SEGMENTS_PER_FILE'] = config.getint('segments', 'segments_per_file')

    cfg['DATA_DIR'] = config.get('data', 'data_dir')

    cfg['CLASSES'] = [
        config.get('classes', 'class0'),
        config.get('classes', 'class1'),
    ]

    cfg['EPOCHS'] = config.getint('training', 'epochs')
    cfg['EARLY_STOPPING_PATIENCE'] = config.getint('training', 'early_stopping_patience')
    cfg['EARLY_STOPPING_MIN_DELTA'] = config.getfloat('training', 'early_stopping_min_delta')
    cfg['WAKEWORD_RATIO'] = config.getfloat('training', 'wakeword_ratio')

    return cfg


def extract_wakeword_segment(spec, seg_frames):
    energy = spec.mean(axis=0)
    peak = np.argmax(energy)

    start = peak - seg_frames // 2
    if start < 0:
        start = 0

    end = start + seg_frames
    if end > spec.shape[1]:
        end = spec.shape[1]
        start = end - seg_frames

    return spec[:, start:end]


def load_training_data(cfg):
    DATA_DIR = cfg['DATA_DIR']
    CLASSES = cfg['CLASSES']

    SR = cfg['SR']
    N_MELS = cfg['N_MELS']
    N_FFT = cfg['N_FFT']
    HOP = cfg['HOP']

    SEGMENT_FRAMES = cfg['SEGMENT_FRAMES']

    file_paths = []
    labels = []

    for class_idx, cname in enumerate(CLASSES):
        class_dir = os.path.join(DATA_DIR, cname)
        wav_files = sorted(glob.glob(os.path.join(class_dir, '**/*.wav'), recursive=True))

        for fn in wav_files:
            file_paths.append(fn)
            labels.append(class_idx)

    segments = []
    seg_labels = []

    for fn, label in zip(file_paths, labels):
        spec = compute_mel_spectrogram(fn, SR, N_FFT, HOP, N_MELS)
        T = spec.shape[1]

        if T < SEGMENT_FRAMES:
            pad = SEGMENT_FRAMES - T
            spec = np.pad(spec, ((0, 0), (0, pad)), mode='constant')
            T = SEGMENT_FRAMES

        if label == 1:
            seg = extract_wakeword_segment(spec, SEGMENT_FRAMES)
            segments.append(seg)
            seg_labels.append(1)

        else:
            # Chose random segment for non-wakeword
            max_offset = T - SEGMENT_FRAMES
            start = np.random.randint(0, max_offset + 1)
            seg = spec[:, start:start + SEGMENT_FRAMES]
            segments.append(seg)
            seg_labels.append(0)

    return np.array(segments), np.array(seg_labels)


if __name__ == '__main__':
    cfg = load_config()
    X, y = load_training_data(cfg)

    print('X shape:', X.shape)
    print('y shape:', y.shape)
    print('Class distribution:', {c: sum(y == i) for i, c in enumerate(cfg['CLASSES'])})
