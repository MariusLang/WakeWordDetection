# Detection and Inference

## Overview

Two backends (PyTorch for Mac/CUDA/CPU, Hailo for Raspberry Pi) each with file inference and continuous detection modes.

## PyTorch Detection

### File Inference

```bash
python -m detection.pytorch.pytorch_inference path/to/audio.wav --model experiments/wakeword_model_TIMESTAMP/ --threshold 0.2
```

**Arguments:** `audio_file` (required), `--model`, `--threshold`

### Continuous Detection

```bash
python -m detection.pytorch.pytorch_detector --model experiments/wakeword_model_TIMESTAMP/ --threshold 0.2 --cooldown 2.0 --detection-dir detections
```

**Arguments:** `--model` (required), `--model-name`, `--threshold`, `--cooldown`, `--no-save`, `--detection-dir`, `--device`

## Hailo Detection

### File Inference

```bash
python3 -m detection.hailo.hailo_inference path/to/audio.wav --hef wakeword.hef --threshold 0.2
```

### Continuous Detection

```bash
python3 -m detection.hailo.hailo_detector --hef wakeword.hef --threshold 0.2 --cooldown 2.0
```

## Detection Logic

Audio is split into overlapping ~1-second segments. Each segment is classified as wake word or not. Detection triggers when the ratio of wake word segments exceeds the threshold (default 0.2 = 20%).

**Threshold tuning:** Lower (0.1) = more sensitive; Higher (0.3+) = fewer false positives.

## Manual Marking

Press `s` during continuous detection to mark missed wake words. Saves 3-second clips for retraining. Files saved as `manual_TIMESTAMP_####.wav`.

## Troubleshooting

- **No audio:** Check microphone permissions and device selection
- **High false positives:** Increase threshold, add false positives to training data
- **Missed detections:** Decrease threshold, use manual marking to collect samples

## Next Steps

- [Deployment](deployment.md) - Deploy to Raspberry Pi
- [Model Compilation](model-compilation.md) - Compile for Hailo accelerator
