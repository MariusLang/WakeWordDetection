import os
import urllib.request
import tarfile
import shutil

DATA_DIR = "data/non_wakeword"
URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
ARCHIVE = "speech_commands.tar.gz"

NON_WAKE_CLASSES = [
    "yes", "no", "up", "down", "left", "right",
    "on", "off", "go", "stop",
    "one", "two", "three", "four", "five",
    "bed", "cat", "dog", "house",
    "_background_noise_"
]


def download_dataset():
    if not os.path.isfile(ARCHIVE):
        print("[INFO] Lade Speech Commands herunter...")
        urllib.request.urlretrieve(URL, ARCHIVE)
        print("[INFO] Download abgeschlossen.")
    else:
        print("[INFO] Archiv bereits vorhanden.")


def extract_nonwakeword():
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)

    with tarfile.open(ARCHIVE, "r:gz") as tar:
        members = tar.getmembers()

        print("[INFO] Extrahiere Non-Wakeword Klassen...")

        for m in members:
            norm = os.path.normpath(m.name)  # -> z.B. "yes/001.wav"
            top_dir = norm.split(os.sep)[0]  # -> "yes"

            if top_dir in NON_WAKE_CLASSES:
                # Wir extrahieren in einen temp-Ordner:
                m.path = m.name  # wichtig für tarfile Sicherheit
                tar.extract(m, DATA_DIR)

    print("[INFO] Extraktion abgeschlossen.")


def flatten_structure():
    print("[INFO] Bereinige Ordner...")

    for cls in NON_WAKE_CLASSES:
        cls_path = os.path.join(DATA_DIR, cls)

        if not os.path.isdir(cls_path):
            continue

        # Manche Tar-Versionen extrahieren als:
        # data/non_wakeword/yes/yes/*.wav
        nested = os.path.join(cls_path, cls)
        if os.path.isdir(nested):
            for fn in os.listdir(nested):
                shutil.move(os.path.join(nested, fn), cls_path)
            os.rmdir(nested)

    print("[INFO] Fertig!")


if __name__ == "__main__":
    download_dataset()
    extract_nonwakeword()
    flatten_structure()
    print("[DONE] Non-WakeWord Daten bereit.")
