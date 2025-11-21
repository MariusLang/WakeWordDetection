import sounddevice as sd
import wave
import datetime

samplerate = 16000  # Hz
channels = 1

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
                wf.writeframes(data.tobytes())

    except KeyboardInterrupt:
        print('\nStopped.')

    finally:
        wf.close()
        print('File saved.')
