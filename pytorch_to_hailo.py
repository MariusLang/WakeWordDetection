import torch
import argparse
import subprocess
import json
import os

from model.model_registry import get_model


def export_to_onnx(model, onnx_path, input_shape):
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=17,
        dynamic_axes=None
    )
    print(f'Exported ONNX: {onnx_path}')


def compile_to_hef(onnx_path, hef_path, hw_arch='hailo8'):
    cmd = [
        'hailo_model_compiler',
        '--model', onnx_path,
        '--hw_arch', hw_arch,
        '--output', hef_path
    ]

    print('Compiling to HEF...')
    try:
        subprocess.run(cmd, check=True)
        print(f'HEF generated: {hef_path}')
    except FileNotFoundError:
        print('hailo_model_compiler not found. Install Hailo SDK.')
    except subprocess.CalledProcessError as e:
        print('Compiler failed:')
        print(e)


def main():
    parser = argparse.ArgumentParser(description='WakeWord model to ONNX to HEF exporter')

    parser.add_argument('--pt', required=True, help='Path to the .pt model file')
    parser.add_argument('--onnx', default='wakeword.onnx', help='Output ONNX filename')
    parser.add_argument('--hef', default='wakeword.hef', help='Output HEF filename')
    parser.add_argument('--shape', nargs='+', type=int, required=True,
                        help='Input shape: e.g. 1 1 40 32')
    parser.add_argument('--hw', default='hailo8', help='Hailo hardware: hailo8 or hailo8l')
    parser.add_argument('--skip_hef', action='store_true', help='Only export ONNX')
    parser.add_argument('--model', default=None, help='Model architecture: cnn or crnn (auto-detected from config.json if not specified)')

    args = parser.parse_args()

    C, H, W = args.shape[1:]
    num_classes = 2  # Default

    # Auto-detect model architecture from config.json if not specified
    model_arch = args.model
    if model_arch is None:
        config_path = os.path.join(os.path.dirname(args.pt), 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                model_arch = config.get('model_architecture', 'cnn')
                num_classes = config.get('num_classes', 2)
                print(f'Auto-detected model architecture: {model_arch}')
        else:
            model_arch = 'cnn'
            print(f'No config.json found, defaulting to "cnn" architecture')

    # Create model using the registry
    model = get_model(model_arch, input_shape=(C, H, W), num_classes=num_classes)
    model.load_state_dict(torch.load(args.pt, map_location='cpu'))
    model.eval()

    export_to_onnx(model, args.onnx, args.shape)

    if not args.skip_hef:
        compile_to_hef(args.onnx, args.hef, args.hw)


if __name__ == '__main__':
    main()
