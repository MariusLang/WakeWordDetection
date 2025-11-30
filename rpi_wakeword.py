import sys
import numpy as np
from hailo_platform import VDevice, HEF, InputVStreams, OutputVStreams, InputVStreamParams, OutputVStreamParams, \
    FormatType

from utils.data_loader import load_config
from utils.audio_processing import preprocess_audio_file


def predict_wakeword(fn, hef_path='wakeword.hef'):
    """Predict if audio file contains wake word using HEF model."""
    cfg = load_config()

    print(f'Loading model: {hef_path}')

    # Preprocess audio into segments
    X = preprocess_audio_file(fn, cfg)
    print(f'Created {X.shape[0]} segments from file.')
    print(f'Segment shape: {X.shape[1:]}')

    # Initialize Hailo device
    with VDevice() as vdevice:
        # Load HEF
        hef = HEF(hef_path)

        # Configure network
        network_groups = vdevice.configure(hef)
        network_group = network_groups[0]

        # Create vstream parameters
        # Use quantized=False for both to send/receive FLOAT32
        # HailoRT will handle quantization/dequantization internally
        input_vstreams_params = InputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=FormatType.FLOAT32
        )
        output_vstreams_params = OutputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=FormatType.FLOAT32
        )

        # Create input and output vstreams
        input_vstream_infos = hef.get_input_vstream_infos()
        output_vstream_infos = hef.get_output_vstream_infos()

        print(f'Input vstream: {input_vstream_infos[0].name}')
        print(f'Output vstream: {output_vstream_infos[0].name}')

        with InputVStreams(network_group, input_vstreams_params) as input_vstreams:
            with OutputVStreams(network_group, output_vstreams_params) as output_vstreams:

                # Get the actual vstream objects using iterator
                input_vstream_list = list(input_vstreams)
                output_vstream_list = list(output_vstreams)

                # Assuming single input/output
                input_vstream = input_vstream_list[0]
                output_vstream = output_vstream_list[0]

                print(f'\nRunning inference on {len(X)} segments...')

                # Activate the network group using context manager
                with network_group.activate():

                    # Process each segment individually
                    all_probs = []
                    for i, segment in enumerate(X):
                        # Add batch dimension: (40, 100, 1) -> (1, 40, 100, 1) for HWC format
                        segment_batched = np.expand_dims(segment, axis=0)
                        segment_batched = np.ascontiguousarray(segment_batched, dtype=np.float32)

                        # Send batched array (batch_size=1)
                        input_vstream.send(segment_batched)

                        # Receive output
                        output_data = output_vstream.recv()

                        # Output is FLOAT32 logits, apply softmax
                        exp_out = np.exp(output_data - np.max(output_data))
                        probs = exp_out / np.sum(exp_out)

                        all_probs.append(probs)

                        if (i + 1) % 10 == 0 or i == 0:
                            print(f'  Processed {i + 1}/{len(X)} segments')

                # Convert to numpy array
                probs_np = np.array(all_probs)
                wake_probs = probs_np[:, 1]

                print('\n--- WakeWord Detection ---')
                print('Max wakeword prob:', float(np.max(wake_probs)))

                preds = np.argmax(probs_np, axis=1)
                n_wake = np.sum(preds == 1)
                n_non = np.sum(preds == 0)

                print(f'Frames predicted as wakeword: {n_wake}/{len(preds)}')

                ratio = n_wake / len(preds)
                detected = ratio > 0.2

                if detected:
                    print('✅ WAKEWORD DETECTED')
                else:
                    print('❌ No wakeword')

                return detected


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage:\n  python rpi_wakeword.py path/to/audio.wav [path/to/model.hef]')
        sys.exit(1)

    fn = sys.argv[1]
    hef_path = sys.argv[2] if len(sys.argv) > 2 else 'wakeword.hef'

    predict_wakeword(fn, hef_path)