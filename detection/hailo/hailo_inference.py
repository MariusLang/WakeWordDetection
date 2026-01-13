import argparse
import numpy as np

from hailo_platform import (
    VDevice, HEF,
    InputVStreams, OutputVStreams,
    InputVStreamParams, OutputVStreamParams,
    FormatType
)

from utils.data_loader import load_config
from utils.audio_processing import preprocess_audio_file


def predict_wakeword(audio_file: str, hef_path: str = 'wakeword.hef', threshold: float = 0.2) -> bool:
    """
    Predict if audio file contains wake word using HEF model.

    Args:
        audio_file: Path to audio WAV file
        hef_path: Path to HEF model file
        threshold: Detection threshold (default: 0.2)

    Returns:
        bool: True if wake word detected
    """
    cfg = load_config()

    print(f'Loading model: {hef_path}')

    X = preprocess_audio_file(audio_file, cfg)
    print(f'Created {X.shape[0]} segments from file.')
    print(f'Segment shape: {X.shape[1:]}')

    with VDevice() as vdevice:
        hef = HEF(hef_path)

        network_groups = vdevice.configure(hef)
        network_group = network_groups[0]

        input_vstreams_params = InputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=FormatType.FLOAT32
        )
        output_vstreams_params = OutputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=FormatType.FLOAT32
        )

        input_vstream_infos = hef.get_input_vstream_infos()
        output_vstream_infos = hef.get_output_vstream_infos()

        print(f'Input vstream: {input_vstream_infos[0].name}')
        print(f'Output vstream: {output_vstream_infos[0].name}')

        with InputVStreams(network_group, input_vstreams_params) as input_vstreams:
            with OutputVStreams(network_group, output_vstreams_params) as output_vstreams:

                input_vstream = list(input_vstreams)[0]
                output_vstream = list(output_vstreams)[0]

                print(f'\nRunning inference on {len(X)} segments...')

                with network_group.activate():
                    all_probs = []

                    for i, segment in enumerate(X):
                        segment_batched = np.expand_dims(segment, axis=0)
                        segment_batched = np.ascontiguousarray(segment_batched, dtype=np.float32)

                        input_vstream.send(segment_batched)
                        output_data = output_vstream.recv()

                        # Apply softmax
                        exp_out = np.exp(output_data - np.max(output_data))
                        probs = exp_out / np.sum(exp_out)

                        all_probs.append(probs)

                        if (i + 1) % 10 == 0 or i == 0:
                            print(f'  Processed {i + 1}/{len(X)} segments')

                probs_np = np.array(all_probs)
                wake_probs = probs_np[:, 1]

                print('\n--- WakeWord Detection ---')
                print(f'Max wakeword prob: {float(np.max(wake_probs)):.4f}')

                preds = np.argmax(probs_np, axis=1)
                n_wake = np.sum(preds == 1)

                print(f'Frames predicted as wakeword: {n_wake}/{len(preds)}')

                ratio = n_wake / len(preds)
                detected = ratio > threshold

                if detected:
                    print('WAKEWORD DETECTED')
                else:
                    print('No wakeword')

                return detected


def main():
    parser = argparse.ArgumentParser(
        description='Run inference on audio file with Hailo HEF model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('audio_file', type=str, help='Path to audio WAV file')
    parser.add_argument('--hef', type=str, default='wakeword.hef',
                        help='Path to HEF model file (default: wakeword.hef)')
    parser.add_argument('--threshold', type=float, default=0.2,
                        help='Detection threshold (default: 0.2)')

    args = parser.parse_args()

    predict_wakeword(args.audio_file, args.hef, args.threshold)


if __name__ == '__main__':
    main()
