import os
import hashlib
import numpy as np
import soundfile as sf


class SampleSaver:
    def __init__(self, sample_rate, folder_name="datasets/HeyPiSamples"):
        self.sample_rate = sample_rate
        self.folder = folder_name
        self.files_count = 0

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
            print(f"Created folder: {self.folder}")
        else:
            self.files_count = sum(1 for entry in os.scandir(folder_name) if entry.is_file())
            print(f"Founded {self.files_count} audio files.")

    def save(self, audio: np.ndarray):
        if len(audio) == 0:
            print("Audio buffer is empty, nothing to save.")
            return None
        audio = audio.astype(np.float32)
        hash_str = hashlib.md5(audio.tobytes()).hexdigest()
        filepath = os.path.join(self.folder, f"hey_pi_{hash_str}.wav")
        sf.write(filepath, audio, self.sample_rate)

        print(f"Saved: {filepath}\t total files count: {self.files_count}")
        self.files_count += 1
        return filepath

import sounddevice as sd
import numpy as np
import queue
import librosa
from random import randint


SAMPLE_RATE = 16000
WINDOW_SIZE = 1.0

WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SIZE)

audio_queue = queue.Queue()
audio_buffer = np.array([], dtype=np.float32)


def callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy().flatten())

word_detected_flag = False

if __name__ == '__main__':

    sample_saver = SampleSaver(16000)

    with sd.InputStream(channels=1, callback=callback) as stream:
        print("Start recording...")
        sr = stream.samplerate
        print("Actual mic sample rate:", sr)
        try:
            while True:
                new_audio = librosa.resample(audio_queue.get(), orig_sr=int(sr), target_sr=SAMPLE_RATE)
                if new_audio is None:
                    continue
                if not word_detected_flag and len(audio_buffer) >= WINDOW_SAMPLES:
                    avg_loughtness = np.mean(np.abs(audio_buffer))
                    new_audio_loughtness = np.mean(np.abs(new_audio))
                    if (avg_loughtness * 20 < new_audio_loughtness):
                            print("Word Detected")
                            word_detected_flag = True
                            audio_buffer = audio_buffer[-min(len(audio_buffer),randint(0,WINDOW_SAMPLES//4)):] #Randomly cut previous buffer before word
                else:
                    if (len(audio_buffer) >= WINDOW_SAMPLES):
                        word_audio = audio_buffer[:WINDOW_SAMPLES]
                        sd.play(word_audio, SAMPLE_RATE)
                        sample_saver.save(word_audio)
                        word_detected_flag = False
                audio_buffer = np.concatenate((audio_buffer, new_audio))
                if len(audio_buffer) >= WINDOW_SAMPLES * 2:
                    audio_buffer = audio_buffer[WINDOW_SAMPLES:]
        except KeyboardInterrupt:
            print("Exiting from keyboard interrupt")
