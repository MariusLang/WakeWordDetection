import sys
import os
import queue
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
from abc import ABC, abstractmethod
from datetime import datetime

from utils.data_loader import load_config
from utils.audio_processing import preprocess_audio_chunk


class BaseWakeWordDetector(ABC):
    """
    Base class for continuous wake word detection.
    """

    def __init__(
            self,
            sample_rate: int = 16000,
            chunk_duration: float = 1.5,
            detection_threshold: float = 0.2,
            cooldown_seconds: float = 2.0,
            save_detections: bool = True,
            detection_dir: str = 'detections',
            manual_recording_duration: float = 3.0,
    ):
        self.cfg = load_config()

        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)

        self.detection_threshold = detection_threshold
        self.cooldown_seconds = cooldown_seconds
        self.last_detection_time = None

        self.save_detections = save_detections
        self.detection_dir = detection_dir

        self.manual_recording_duration = manual_recording_duration
        self.manual_recording_samples = int(sample_rate * manual_recording_duration)

        self.audio_queue = queue.Queue()
        self.running = False
        self.detection_count = 0
        self.manual_detection_count = 0
        self.manual_save_flag = False
        self.audio_history = np.array([], dtype=np.float32)

        if self.save_detections:
            os.makedirs(self.detection_dir, exist_ok=True)
            print(f'Detection recordings will be saved to: {self.detection_dir}/')

        self._print_init_info()

    def _print_init_info(self):
        print('Initialized Continuous Wake Word Detector')
        print(f'Sample rate: {self.sample_rate} Hz')
        print(f'Chunk duration: {self.chunk_duration}s')
        print(f'Manual recording duration: {self.manual_recording_duration}s')
        print(f'Detection threshold: {self.detection_threshold}')
        print(f'Cooldown: {self.cooldown_seconds}s')

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f'Audio status: {status}')
        self.audio_queue.put(indata.copy())

    def should_detect(self) -> bool:
        if self.last_detection_time is None:
            return True
        elapsed = (datetime.now() - self.last_detection_time).total_seconds()
        return elapsed >= self.cooldown_seconds

    def keyboard_listener(self):
        if not sys.stdin.isatty():
            print('Keyboard input disabled (no TTY detected). Run in a Terminal for "s" hotkey.')
            return

        import select
        import tty
        import termios

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while self.running:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    if key == 's':
                        self.manual_save_flag = True
                    elif key == '\x03':  # Ctrl+C
                        self.running = False
                        break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def save_detection_audio(self, audio_chunk, max_prob: float, ratio: float, manual: bool = False):
        if not self.save_detections:
            return None

        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            if manual:
                self.manual_detection_count += 1
                filename = f'manual_{timestamp}_{self.manual_detection_count:04d}.wav'
            else:
                self.detection_count += 1
                filename = (
                    f'detection_{timestamp}_'
                    f'prob{max_prob:.3f}_ratio{ratio:.3f}_'
                    f'{self.detection_count:04d}.wav'
                )

            filepath = os.path.join(self.detection_dir, filename)
            sf.write(filepath, audio_chunk, self.sample_rate)
            return filepath
        except Exception as e:
            print(f'\nError saving detection audio: {e}')
            return None

    @abstractmethod
    def _load_model(self):
        """
        Load the model.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _run_inference(self, segments) -> np.ndarray:
        """
        Run inference on preprocessed segments. Returns probabilities array.

        Args:
            segments: Preprocessed audio segments

        Returns:
            np.ndarray: Probability array of shape (n_segments, n_classes)
        """
        pass

    def process_audio_chunk(self, audio_chunk) -> tuple[bool, float, float]:
        X = preprocess_audio_chunk(audio_chunk, self.cfg)

        if X is None or len(X) == 0:
            return False, 0.0, 0.0

        probs_np = self._run_inference(X)
        wake_probs = probs_np[:, 1]

        max_prob = float(np.max(wake_probs))
        preds = np.argmax(probs_np, axis=1)
        ratio = float(np.mean(preds == 1))

        detected = ratio > self.detection_threshold
        return detected, max_prob, ratio

    def _print_summary(self):
        if self.save_detections and (self.detection_count > 0 or self.manual_detection_count > 0):
            print(f'\nSession Summary:')
            print(f'   Auto detections: {self.detection_count}')
            print(f'   Manual marks: {self.manual_detection_count}')
            print(f'   Recordings saved in: {self.detection_dir}/')

    def run(self):
        """
        Start continuous detection.

        Can be overridden for platform-specific setup.
        """
        self.running = True

        keyboard_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        keyboard_thread.start()

        with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.audio_callback,
                blocksize=int(self.sample_rate * 0.1),
        ):
            print('\n' + '=' * 60)
            print('LISTENING FOR WAKE WORD')
            print('Press "s" to manually mark a wakeword (Terminal only)')
            print('Press Ctrl+C to stop')
            print('=' * 60 + '\n')

            audio_buffer = np.array([], dtype=np.float32)

            try:
                while self.running:
                    try:
                        chunk = self.audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    chunk_flat = chunk.flatten()
                    audio_buffer = np.append(audio_buffer, chunk_flat)

                    # Update audio history for manual marking
                    self.audio_history = np.append(self.audio_history, chunk_flat)
                    if len(self.audio_history) > self.manual_recording_samples:
                        self.audio_history = self.audio_history[-self.manual_recording_samples:]

                    # Handle manual mark
                    if self.manual_save_flag:
                        self.manual_save_flag = False
                        saved = self.save_detection_audio(self.audio_history, 0.0, 0.0, manual=True)
                        print(f'[{datetime.now().strftime("%H:%M:%S")}] MANUAL MARK ({saved})')

                    # Process when enough audio collected
                    if len(audio_buffer) >= self.chunk_samples:
                        audio_chunk = audio_buffer[:self.chunk_samples]
                        audio_buffer = audio_buffer[self.chunk_samples // 2:]  # 50% overlap

                        detected, max_prob, ratio = self.process_audio_chunk(audio_chunk)
                        timestamp = datetime.now().strftime('%H:%M:%S')

                        if detected and self.should_detect():
                            saved = self.save_detection_audio(audio_chunk, max_prob, ratio)
                            print(
                                f'[{timestamp}] WAKE WORD DETECTED '
                                f'(prob={max_prob:.3f}, ratio={ratio:.3f}) '
                                f'{saved}'
                            )
                            self.last_detection_time = datetime.now()
                        else:
                            print(
                                f'[{timestamp}] Listening '
                                f'(prob={max_prob:.3f}, ratio={ratio:.3f})',
                                end='\r',
                            )

            except KeyboardInterrupt:
                print('\nStopping detection')
                self.running = False
                self._print_summary()
