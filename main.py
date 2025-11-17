import configparser
import os
import sys


def load_config(path: str = 'config.ini'):
    config = configparser.ConfigParser()

    if not os.path.isfile(path):
        print(f'Config file not found at {path}')
        sys.exit(1)

    config.read(path)

    SR = config.getint('audio', 'sr')
    N_MELS = config.getint('audio', 'n_mels')
    N_FFT = config.getint('audio', 'n_fft')
    HOP = config.getint('audio', 'hop')

    SEGMENT_FRAMES = config.getint('segments', 'segment_frames')
    SEGMENTS_PER_FILE = config.getint('segments', 'segments_per_file')

    DATA_DIR = config.get('data', 'data_dir')

    CLASSES = [
        config.get('classes', 'class0'),
        config.get('classes', 'class1')
    ]

    return {
        'SR': SR,
        'N_MELS': N_MELS,
        'N_FFT': N_FFT,
        'HOP': HOP,
        'SEGMENT_FRAMES': SEGMENT_FRAMES,
        'SEGMENTS_PER_FILE': SEGMENTS_PER_FILE,
        'DATA_DIR': DATA_DIR,
        'CLASSES': CLASSES
    }


if __name__ == '__main__':
    cfg = load_config()
    print(cfg['SR'])
