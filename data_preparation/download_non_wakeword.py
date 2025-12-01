import os
import urllib.request
import tarfile
import shutil

DATA_DIR = '../data/non_wakeword'
URL = 'https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
ARCHIVE = 'speech_commands.tar.gz'

NON_WAKE_CLASSES = [
    'yes', 'no', 'up', 'down', 'left', 'right',
    'on', 'off', 'go', 'stop',
    'one', 'two', 'three', 'four', 'five',
    'bed', 'cat', 'dog', 'house',
    '_background_noise_'
]


def download_dataset():
    if not os.path.isfile(ARCHIVE):
        print('Load speech commands dataset from tensorflow.org')
        urllib.request.urlretrieve(URL, ARCHIVE)
        print('Download successful.')
    else:
        print('Archive already exists.')


def extract_nonwakeword():
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)

    with tarfile.open(ARCHIVE, 'r:gz') as tar:
        members = tar.getmembers()

        print('Extracting files...')

        for m in members:
            norm = os.path.normpath(m.name)
            top_dir = norm.split(os.sep)[0]

            if top_dir in NON_WAKE_CLASSES:
                m.path = m.name
                tar.extract(m, DATA_DIR)

    print('Extraction complete.')


def flatten_structure():
    for cls in NON_WAKE_CLASSES:
        cls_path = os.path.join(DATA_DIR, cls)

        if not os.path.isdir(cls_path):
            continue

        nested = os.path.join(cls_path, cls)
        if os.path.isdir(nested):
            for fn in os.listdir(nested):
                shutil.move(os.path.join(nested, fn), cls_path)
            os.rmdir(nested)


if __name__ == '__main__':
    download_dataset()
    extract_nonwakeword()
    flatten_structure()
    print('Successfully downloaded non-wake word data.')
