import argparse
import numpy as np
import sounddevice as sd
import threading
from datetime import datetime

from hailo_platform import (
    VDevice, HEF,
    InputVStreams, OutputVStreams,
    InputVStreamParams, OutputVStreamParams,
    FormatType
)

from detection.base_detector import BaseWakeWordDetector


class HailoWakeWordDetector(BaseWakeWordDetector):
    """
    Wake word detector using Hailo accelerator (Raspberry Pi).
    """

    def __init__(
        self,
        hef_path: str = 'wakeword.hef',
        sample_rate: int = 16000,
        chunk_duration: float = 1.5,
        detection_threshold: float = 0.2,
        cooldown_seconds: float = 2.0,
        save_detections: bool = True,
        detection_dir: str = 'detections',
        manual_recording_duration: float = 3.0,
    ):
        self.hef_path = hef_path

        super().__init__(
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            detection_threshold=detection_threshold,
            cooldown_seconds=cooldown_seconds,
            save_detections=save_detections,
            detection_dir=detection_dir,
            manual_recording_duration=manual_recording_duration,
        )

        # Hailo resources are initialized in run() due to context manager requirements
        self._input_vstream = None
        self._output_vstream = None

    def _load_model(self):
        """
        Model loading happens in run() for Hailo due to context managers.
        """
        pass

    def _run_inference(self, segments) -> np.ndarray:
        if self._input_vstream is None or self._output_vstream is None:
            raise RuntimeError('Hailo streams not initialized. Call run() first.')

        all_probs = []

        for segment in segments:
            segment_batched = np.expand_dims(segment, axis=0)
            segment_batched = np.ascontiguousarray(segment_batched, dtype=np.float32)

            self._input_vstream.send(segment_batched)
            output_data = self._output_vstream.recv()

            # Apply softmax
            exp_out = np.exp(output_data - np.max(output_data))
            probs = exp_out / np.sum(exp_out)
            all_probs.append(probs)

        return np.array(all_probs)

    def run(self):
        self.running = True

        print(f'\nLoading model: {self.hef_path}')
        print('Starting audio stream...')

        with VDevice() as vdevice:
            hef = HEF(self.hef_path)
            network_groups = vdevice.configure(hef)
            network_group = network_groups[0]

            input_vstreams_params = InputVStreamParams.make_from_network_group(
                network_group, quantized=False, format_type=FormatType.FLOAT32
            )
            output_vstreams_params = OutputVStreamParams.make_from_network_group(
                network_group, quantized=False, format_type=FormatType.FLOAT32
            )

            with InputVStreams(network_group, input_vstreams_params) as input_vstreams:
                with OutputVStreams(network_group, output_vstreams_params) as output_vstreams:

                    self._input_vstream = list(input_vstreams)[0]
                    self._output_vstream = list(output_vstreams)[0]

                    with network_group.activate():
                        keyboard_thread = threading.Thread(
                            target=self.keyboard_listener, daemon=True
                        )
                        keyboard_thread.start()

                        with sd.InputStream(
                            samplerate=self.sample_rate,
                            channels=1,
                            callback=self.audio_callback,
                            blocksize=int(self.sample_rate * 0.1),
                        ):
                            print('\n' + '=' * 60)
                            print('LISTENING FOR WAKE WORD')
                            print('Press "s" to manually mark a wakeword')
                            print('Press Ctrl+C to stop')
                            print('=' * 60 + '\n')

                            audio_buffer = np.array([], dtype=np.float32)

                            try:
                                while self.running:
                                    try:
                                        chunk = self.audio_queue.get(timeout=0.1)
                                        chunk_flat = chunk.flatten()
                                        audio_buffer = np.append(audio_buffer, chunk_flat)

                                        self.audio_history = np.append(
                                            self.audio_history, chunk_flat
                                        )
                                        if len(self.audio_history) > self.manual_recording_samples:
                                            self.audio_history = self.audio_history[
                                                -self.manual_recording_samples:
                                            ]
                                    except Exception:
                                        continue

                                    if self.manual_save_flag:
                                        self.manual_save_flag = False
                                        timestamp = datetime.now().strftime('%H:%M:%S')
                                        saved_path = self.save_detection_audio(
                                            self.audio_history, 0.0, 0.0, manual=True
                                        )
                                        msg = f'\n[{timestamp}] MANUAL MARK! ({self.manual_recording_duration}s recording)'
                                        if saved_path:
                                            import os
                                            msg += f' - Saved: {os.path.basename(saved_path)}'
                                        print(msg)

                                    if len(audio_buffer) >= self.chunk_samples:
                                        audio_chunk = audio_buffer[:self.chunk_samples]
                                        audio_buffer = audio_buffer[self.chunk_samples // 2:]

                                        detected, max_prob, ratio = self.process_audio_chunk(
                                            audio_chunk
                                        )
                                        timestamp = datetime.now().strftime('%H:%M:%S')

                                        if detected and self.should_detect():
                                            saved_path = self.save_detection_audio(
                                                audio_chunk, max_prob, ratio
                                            )
                                            msg = (
                                                f'[{timestamp}] WAKE WORD DETECTED! '
                                                f'(max_prob: {max_prob:.3f}, ratio: {ratio:.3f})'
                                            )
                                            if saved_path:
                                                import os
                                                msg += f' - Saved: {os.path.basename(saved_path)}'
                                            print(msg)
                                            self.last_detection_time = datetime.now()
                                        else:
                                            print(
                                                f'[{timestamp}] Listening... '
                                                f'(max_prob: {max_prob:.3f}, ratio: {ratio:.3f})',
                                                end='\r',
                                            )

                            except KeyboardInterrupt:
                                print('\n\nStopping detection...')
                                self.running = False
                                self._print_summary()


def main():
    parser = argparse.ArgumentParser(
        description='Continuous wake word detection using Hailo accelerator (Raspberry Pi)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--hef', type=str, default='wakeword.hef',
                        help='Path to HEF model file (default: wakeword.hef)')
    parser.add_argument('--threshold', type=float, default=0.2,
                        help='Detection threshold (default: 0.2)')
    parser.add_argument('--cooldown', type=float, default=2.0,
                        help='Cooldown seconds between detections (default: 2.0)')
    parser.add_argument('--chunk-duration', type=float, default=1.5,
                        help='Audio chunk duration in seconds (default: 1.5)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save detection audio files')
    parser.add_argument('--detection-dir', type=str, default='detections',
                        help='Directory to save detection recordings (default: detections)')

    args = parser.parse_args()

    print('Continuous Wake Word Detection (Hailo)')
    print('=' * 60)

    detector = HailoWakeWordDetector(
        hef_path=args.hef,
        sample_rate=16000,
        chunk_duration=args.chunk_duration,
        detection_threshold=args.threshold,
        cooldown_seconds=args.cooldown,
        save_detections=not args.no_save,
        detection_dir=args.detection_dir,
        manual_recording_duration=3.0,
    )

    detector.run()


if __name__ == '__main__':
    main()
