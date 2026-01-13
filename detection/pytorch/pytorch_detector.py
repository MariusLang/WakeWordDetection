import argparse
import numpy as np
import torch

from detection.base_detector import BaseWakeWordDetector
from model.model_registry import list_models, get_model
from utils.get_device import get_device


def infer_model_name_from_state_dict(state_dict: dict) -> str:
    """
    Infer model architecture from state dict keys.
    """
    keys = state_dict.keys()
    if any(k.startswith('freq_reduce.') for k in keys) or \
       any(k.startswith('temporal.') for k in keys) or \
       any(k.startswith('final_conv.') for k in keys):
        return 'crnn_temporal'
    if any(k.startswith('rnn.') for k in keys):
        return 'crnn'
    return 'cnn'


def extract_state_dict(obj) -> dict:
    """
    Extract state dict from checkpoint or raw state dict.
    """
    if isinstance(obj, dict) and 'state_dict' in obj and isinstance(obj['state_dict'], dict):
        return obj['state_dict']
    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        return obj
    raise TypeError(
        'Unsupported model file format. Expected a plain state_dict or a checkpoint dict containing "state_dict".'
    )


class PyTorchWakeWordDetector(BaseWakeWordDetector):
    """
    Wake word detector using PyTorch for inference (Mac/CUDA/CPU).
    """

    def __init__(
        self,
        model_path: str,
        sample_rate: int = 16000,
        chunk_duration: float = 1.5,
        detection_threshold: float = 0.2,
        cooldown_seconds: float = 2.0,
        save_detections: bool = True,
        detection_dir: str = 'detections',
        manual_recording_duration: float = 3.0,
        device: str | None = None,
        model_name: str | None = None,
        input_shape: tuple | None = None,
        num_classes: int = 2,
    ):
        self.model_path = model_path
        self.device = get_device(device)
        self.num_classes = num_classes
        self.input_shape = input_shape if input_shape is not None else (1, 40, 100)
        self._model_name = model_name

        super().__init__(
            sample_rate=sample_rate,
            chunk_duration=chunk_duration,
            detection_threshold=detection_threshold,
            cooldown_seconds=cooldown_seconds,
            save_detections=save_detections,
            detection_dir=detection_dir,
            manual_recording_duration=manual_recording_duration,
        )

        print(f'Using device: {self.device}')
        print(f'Input shape (C,F,T): {self.input_shape}')

        self.model = self._load_model()
        print()

    def _load_model(self):
        print(f'Loading model file: {self.model_path}')
        obj = torch.load(self.model_path, map_location='cpu')
        state_dict = extract_state_dict(obj)

        chosen_name = self._model_name or infer_model_name_from_state_dict(state_dict)
        available = list_models()
        if chosen_name not in available:
            raise ValueError(f"Model '{chosen_name}' not in registry. Available: {available}")

        print(f'Selected model architecture: {chosen_name}')

        model = get_model(chosen_name, input_shape=self.input_shape, num_classes=self.num_classes)
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()

        print('Model loaded successfully with strict=True')
        return model

    @torch.no_grad()
    def _run_inference(self, segments) -> np.ndarray:
        all_probs = []

        for segment in segments:
            seg = np.asarray(segment, dtype=np.float32)

            # Handle different input formats
            if seg.ndim == 3 and seg.shape[-1] == 1:
                seg = seg[..., 0]  # drop channel dim -> (F,T) or (T,F)

            if seg.ndim != 2:
                raise ValueError(f'Unsupported segment shape after squeeze: {seg.shape}')

            # Ensure correct orientation (F, T)
            if seg.shape[0] != self.input_shape[1]:
                seg = seg.T

            # (F,T) -> (1,1,F,T)
            seg = np.expand_dims(seg, axis=(0, 1))
            x = torch.from_numpy(seg).to(self.device)

            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.detach().cpu().numpy()[0])

        return np.array(all_probs)


def main():
    parser = argparse.ArgumentParser(
        description='Continuous wake word detection using PyTorch (macOS/CUDA/CPU)'
    )

    parser.add_argument('--model', type=str, required=True,
                        help='Path to .pt/.pth state_dict or checkpoint')
    parser.add_argument('--model-name', type=str, default=None,
                        help='Force architecture: one of ' + ', '.join(list_models()))
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--cooldown', type=float, default=2.0)
    parser.add_argument('--chunk-duration', type=float, default=1.5)
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--detection-dir', type=str, default='detections')
    parser.add_argument('--device', type=str, default=None,
                        help='cpu | mps | cuda')

    # Input shape: (C,F,T) without batch
    parser.add_argument('--channels', type=int, default=1)
    parser.add_argument('--n-mels', type=int, default=40)
    parser.add_argument('--n-frames', type=int, default=100)

    args = parser.parse_args()

    detector = PyTorchWakeWordDetector(
        model_path=args.model,
        chunk_duration=args.chunk_duration,
        detection_threshold=args.threshold,
        cooldown_seconds=args.cooldown,
        save_detections=not args.no_save,
        detection_dir=args.detection_dir,
        device=args.device,
        model_name=args.model_name,
        input_shape=(args.channels, args.n_mels, args.n_frames),
        num_classes=2,
    )

    detector.run()


if __name__ == '__main__':
    main()
