import sys
import numpy as np
import sounddevice as sd
import queue
import threading
from datetime import datetime
from hailo_platform import VDevice, HEF, InputVStreams, OutputVStreams, InputVStreamParams, OutputVStreamParams, \
    FormatType

from utils.data_loader import load_config
from utils.audio_processing import compute_mel_spectrogram_from_audio


def normalize_segments(segments):
    """Normalize each segment individually to match training preprocessing."""
    from sklearn.preprocessing import StandardScaler
    segments_norm = np.zeros_like(segments)
    for i in range(len(segments)):
        scaler = StandardScaler()
        segments_norm[i] = scaler.fit_transform(segments[i])
    return segments_norm


def preprocess_audio_chunk(audio_data, cfg):
    """Preprocess audio chunk into segments for inference."""
    SR = cfg['SR']
    N_MELS = cfg['N_MELS']
    N_FFT = cfg['N_FFT']
    HOP = cfg['HOP']
    SEGMENT_FRAMES = cfg['SEGMENT_FRAMES']

    # Compute mel spectrogram from audio array
    spec = compute_mel_spectrogram_from_audio(audio_data, SR, N_FFT, HOP, N_MELS)

    if spec.shape[1] < SEGMENT_FRAMES:
        pad = SEGMENT_FRAMES - spec.shape[1]
        spec = np.pad(spec, ((0, 0), (0, pad)), mode='constant')

    segments = []
    for start in range(0, spec.shape[1] - SEGMENT_FRAMES + 1, 10):
        seg = spec[:, start:start + SEGMENT_FRAMES]
        segments.append(seg)

    if len(segments) == 0:
        return None

    segments = np.array(segments)
    segments_norm = normalize_segments(segments)

    # Add channel dimension for Hailo (HWC format)
    segments_norm = np.expand_dims(segments_norm, axis=-1)

    return segments_norm.astype(np.float32)


class ContinuousWakeWordDetector:
    """Continuous wake word detector using microphone input."""

    def __init__(self, hef_path='wakeword.hef', sample_rate=16000, chunk_duration=1.5,
                 detection_threshold=0.2, cooldown_seconds=2.0):
        self.hef_path = hef_path
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.detection_threshold = detection_threshold
        self.cooldown_seconds = cooldown_seconds
        self.last_detection_time = None

        self.cfg = load_config()
        self.audio_queue = queue.Queue()
        self.running = False

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
                                            print(f'[{timestamp}] ✅ WAKE WORD DETECTED! '
                                                  f'(max_prob: {max_prob:.3f}, ratio: {ratio:.3f})')
                                            self.last_detection_time = datetime.now()
                                        else:
                                            # Print a dot to show it's listening (optional)
                                            print(f'[{timestamp}] Listening... '
                                                  f'(max_prob: {max_prob:.3f}, ratio: {ratio:.3f})',
                                                  end='\r')

                            except KeyboardInterrupt:
                                print('\n\n🛑 Stopping detection...')
                                self.running = False


def main():
    """Main entry point."""
    # Parse arguments
    hef_path = 'wakeword.hef'
    sample_rate = 16000
    chunk_duration = 1.5
    detection_threshold = 0.2
    cooldown = 2.0

    if len(sys.argv) > 1:
        hef_path = sys.argv[1]
    if len(sys.argv) > 2:
        detection_threshold = float(sys.argv[2])
    if len(sys.argv) > 3:
        cooldown = float(sys.argv[3])

    print('Continuous Wake Word Detection')
    print('='*60)

    detector = ContinuousWakeWordDetector(
        hef_path=hef_path,
        sample_rate=sample_rate,
        chunk_duration=chunk_duration,
        detection_threshold=detection_threshold,
        cooldown_seconds=cooldown
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
