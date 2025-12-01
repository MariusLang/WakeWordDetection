import sys
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import queue
from datetime import datetime
from hailo_platform import VDevice, HEF, InputVStreams, OutputVStreams, InputVStreamParams, OutputVStreamParams, \
    FormatType

from utils.data_loader import load_config
from utils.audio_processing import preprocess_audio_chunk


class ContinuousWakeWordDetector:
    """Continuous wake word detector using microphone input."""

    def __init__(self, hef_path='wakeword.hef', sample_rate=16000, chunk_duration=1.5,
                 detection_threshold=0.2, cooldown_seconds=2.0, save_detections=True,
                 detection_dir='detections'):
        self.hef_path = hef_path
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.detection_threshold = detection_threshold
        self.cooldown_seconds = cooldown_seconds
        self.last_detection_time = None
        self.save_detections = save_detections
        self.detection_dir = detection_dir

        self.cfg = load_config()
        self.audio_queue = queue.Queue()
        self.running = False
        self.detection_count = 0

        # Create detection directory if saving is enabled
        if self.save_detections:
            os.makedirs(self.detection_dir, exist_ok=True)
            print(f'Detection recordings will be saved to: {self.detection_dir}/')

        print(f'Initializing Continuous Wake Word Detector')
        print(f'Sample rate: {sample_rate} Hz')
        print(f'Chunk duration: {chunk_duration}s')
        print(f'Detection threshold: {detection_threshold}')
        print(f'Cooldown: {cooldown_seconds}s')

    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream."""
        if status:
            print(f'Audio status: {status}')
        # Copy audio data to queue
        self.audio_queue.put(indata.copy())

    def process_audio_chunk(self, audio_chunk, input_vstream, output_vstream):
        """Process a single audio chunk and return detection result."""
        # Preprocess audio
        X = preprocess_audio_chunk(audio_chunk, self.cfg)

        if X is None or len(X) == 0:
            return False, 0.0

        # Process each segment
        all_probs = []
        for segment in X:
            # Add batch dimension
            segment_batched = np.expand_dims(segment, axis=0)
            segment_batched = np.ascontiguousarray(segment_batched, dtype=np.float32)

            # Send and receive
            input_vstream.send(segment_batched)
            output_data = output_vstream.recv()

            # Apply softmax
            exp_out = np.exp(output_data - np.max(output_data))
            probs = exp_out / np.sum(exp_out)
            all_probs.append(probs)

        # Aggregate results
        probs_np = np.array(all_probs)
        wake_probs = probs_np[:, 1]
        max_prob = float(np.max(wake_probs))

        preds = np.argmax(probs_np, axis=1)
        n_wake = np.sum(preds == 1)
        ratio = n_wake / len(preds)

        detected = ratio > self.detection_threshold

        return detected, max_prob, ratio

    def should_detect(self):
        """Check if enough time has passed since last detection (cooldown)."""
        if self.last_detection_time is None:
            return True
        elapsed = (datetime.now() - self.last_detection_time).total_seconds()
        return elapsed >= self.cooldown_seconds

    def save_detection_audio(self, audio_chunk, max_prob, ratio):
        """Save audio chunk when wakeword is detected."""
        if not self.save_detections:
            return None

        try:
            # Generate filename with timestamp and metadata
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.detection_count += 1

            filename = f'detection_{timestamp}_prob{max_prob:.3f}_ratio{ratio:.3f}_{self.detection_count:04d}.wav'
            filepath = os.path.join(self.detection_dir, filename)

            # Save as WAV file
            sf.write(filepath, audio_chunk, self.sample_rate)

            return filepath
        except Exception as e:
            print(f'\nError saving detection audio: {e}')
            return None

    def run(self):
        """Start continuous detection."""
        self.running = True

        print(f'\nLoading model: {self.hef_path}')
        print('Starting audio stream...')

        # Initialize Hailo device
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

                    input_vstream = list(input_vstreams)[0]
                    output_vstream = list(output_vstreams)[0]

                    with network_group.activate():
                        # Start audio stream
                        with sd.InputStream(samplerate=self.sample_rate,
                                          channels=1,
                                          callback=self.audio_callback,
                                          blocksize=int(self.sample_rate * 0.1)):

                            print('\n' + '='*60)
                            print('🎤 LISTENING FOR WAKE WORD... (Press Ctrl+C to stop)')
                            print('='*60 + '\n')

                            audio_buffer = np.array([], dtype=np.float32)

                            try:
                                while self.running:
                                    # Get audio from queue
                                    try:
                                        chunk = self.audio_queue.get(timeout=0.1)
                                        audio_buffer = np.append(audio_buffer, chunk.flatten())
                                    except queue.Empty:
                                        continue

                                    # Process when we have enough audio
                                    if len(audio_buffer) >= self.chunk_samples:
                                        # Extract chunk to process
                                        audio_chunk = audio_buffer[:self.chunk_samples]
                                        audio_buffer = audio_buffer[self.chunk_samples//2:]  # 50% overlap

                                        # Run detection
                                        detected, max_prob, ratio = self.process_audio_chunk(
                                            audio_chunk, input_vstream, output_vstream
                                        )

                                        timestamp = datetime.now().strftime('%H:%M:%S')

                                        if detected and self.should_detect():
                                            # Save detection audio
                                            saved_path = self.save_detection_audio(audio_chunk, max_prob, ratio)

                                            detection_msg = f'[{timestamp}] ✅ WAKE WORD DETECTED! (max_prob: {max_prob:.3f}, ratio: {ratio:.3f})'
                                            if saved_path:
                                                detection_msg += f' - Saved: {os.path.basename(saved_path)}'
                                            print(detection_msg)

                                            self.last_detection_time = datetime.now()
                                        else:
                                            # Print a dot to show it's listening (optional)
                                            print(f'[{timestamp}] Listening... '
                                                  f'(max_prob: {max_prob:.3f}, ratio: {ratio:.3f})',
                                                  end='\r')

                            except KeyboardInterrupt:
                                print('\n\n🛑 Stopping detection...')
                                self.running = False

                                # Print summary
                                if self.save_detections and self.detection_count > 0:
                                    print(f'\n📊 Session Summary:')
                                    print(f'   Total detections: {self.detection_count}')
                                    print(f'   Recordings saved in: {self.detection_dir}/')
                                    print(f'\n💡 Review recordings to check for false positives')


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Continuous wake word detection with audio recording',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  %(prog)s

  # Specify HEF model and threshold
  %(prog)s --hef wakeword_v2.hef --threshold 0.3

  # Disable saving detection recordings
  %(prog)s --no-save

  # Custom detection directory
  %(prog)s --detection-dir my_detections
        """
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

    print('Continuous Wake Word Detection')
    print('='*60)

    detector = ContinuousWakeWordDetector(
        hef_path=args.hef,
        sample_rate=16000,
        chunk_duration=args.chunk_duration,
        detection_threshold=args.threshold,
        cooldown_seconds=args.cooldown,
        save_detections=not args.no_save,
        detection_dir=args.detection_dir
    )

    try:
        detector.run()
    except Exception as e:
        print(f'\n❌ Error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
