import sounddevice as sd
import numpy as np
import wave
import datetime

samplerate = 44100  # Hz
channels = 1
target_peak = 0.9  # normalize so max amplitude is 90% of full scale


def normalize_int16(audio):
    audio_float = audio.astype(np.float32) / 32768.0

    peak = np.max(np.abs(audio_float))
    if peak > 0:
        audio_float = audio_float * (target_peak / peak)

    return (audio_float * 32767).astype(np.int16)


if __name__ == '__main__':
    filename = f'recording_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.wav'

    print(f'Recording ... Press CTRL+C to stop.\nFile: {filename}')

    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(2)  # 16-bit
    wf.setframerate(samplerate)

    try:
        with sd.InputStream(samplerate=samplerate, channels=channels, dtype='int16') as stream:
            while True:
                data = stream.read(1024)[0]
                normalized = normalize_int16(data)
                wf.writeframes(normalized.tobytes())

    except KeyboardInterrupt:
        print('\nStopped.')

    finally:
        wf.close()
        print('File saved.')
